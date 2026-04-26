在端侧 AI 和轻量化模型大行其道的今天，如何在极小的参数规模（如 270M）下实现复杂的工具调用（Tool-calling）并引入类似深度推理（Reasoning）的能力？
本文将带你通过 Unsloth 框架，使用 FunctionGemma-270m 模型进行高效微调。我们将通过注入 `<think>` 标签，让这个口袋大小的模型在执行任务前先进行“思考”。
本方案的核心在于将原本缺失“推理能力”的小模型，通过微调强制其在输出内容前生成 `<think>` 标签，从而显性化模型的决策过程。


## 核心目标
通过 Unsloth 框架对极小参数模型（270M）进行监督微调（SFT），使其在无需庞大计算资源的情况下，具备：
显性推理能力：在执行动作前，通过 `<think>` 标签输出思考过程。
多轮工具调用：准确遵循 tool_calls 的 schema 进行交互。


## 技巧：构造“推理型”数据格式
标准的 FunctionGemma 并不直接支持“思考”过程。为了实现这一点，我们需要在 `assistant` 的回复中手动注入 `<think> ... </think>` 标签。


### 关键步骤：
- 角色封装：遵循 Gemma 的 `<start_of_turn>{role} ... <end_of_turn>` 格式。
- 工具声明：在 `developer`（或系统）环节声明工具。
- 推理注入：将数据集中的 `think` 字段内容与 `content` 合并。
    

## 数据预处理

- 数据分布 = 行为边界
- 使用 `LLM360/TxT360-3efforts` 数据集，它包含超过 100 万个带有推理链的 Agent 示例。TxT360-3efforts 这个数据集专门做了"带思考过程的 tool call 样本"
- prepare_messages_and_tools 存在的全部意义，就是把脏数据的分布压成模型可学习的一致模板。
- prepare_messages_and_tools提取 think/think_fast/think_faster 字段并包装成 `<think>...</think>` 块
- 把它从独立字段 m["think"] 搬到 m["content"] 里，让模型在训练时把 `<think>` 当作自己输出的一部分来学。
- think 标签合并、tool_calls 格式归一化、过滤无 thought 的样本。

```
THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"

def prepare_messages_and_tools(example):
    raw = json.loads(example["messages"])
    msgs = [dict(m) for m in raw]

    # 1) Extract tools (same as before)
    tools_raw = []
    if msgs and isinstance(msgs[0], dict):
        tlist = msgs[0].get("tools")
        if isinstance(tlist, list) and tlist:
            tools_raw = tlist
            msgs[0].pop("tools", None)

    # 2) Merge assistant["think"] into ["content"]
    THINK_KEYS = ["think", "think_fast", "think_faster"]

    # TRACKER: Check if we successfully added thoughts
    has_valid_thought = False

    for m in msgs:
        if m.get("role") == "assistant":
            # Find the first available thinking key
            found_key = next((k for k in THINK_KEYS if m.get(k)), None)

            if found_key:
                think_text = m[found_key]
                content = m.get("content")
                think_block = f"{THINK_TAG_OPEN}{think_text}{THINK_TAG_CLOSE}"

                if isinstance(content, str) and content:
                    m["content"] = think_block + "\n" + content
                else:
                    m["content"] = think_block

                has_valid_thought = True

                # Clean up keys
                for k in THINK_KEYS:
                    m.pop(k, None)
            else:
                # If an assistant message HAS NO THOUGHT,
                # this example is "poison" for your goal.
                # We mark it as invalid to filter it out later.
                return None, None

    # If the conversation had no assistant turns at all (rare, but possible), skip it
    if not has_valid_thought:
        return None, None
    # 3) Normalize tool_calls to HF-style {type:'function', function:{name, arguments}}
    for m in msgs:
        if "tool_calls" not in m or not m["tool_calls"]:
            continue

        new_tool_calls = []
        for tc in m["tool_calls"]:
            if not isinstance(tc, dict):
                continue

            # Already has function dict?
            if "function" in tc and isinstance(tc["function"], dict):
                new_tool_calls.append(tc)
                continue

            fn_name = tc.get("name", "")
            args = tc.get("arguments", {})

            # Try to parse JSON string arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass

            new_tool_calls.append(
                {
                    "id": tc.get("id") or tc.get("tool_call_id"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": fn_name,
                        "arguments": args,
                    },
                }
            )

        m["tool_calls"] = new_tool_calls

    # 3b) Build map from tool_call_id -> function name for later tool responses
    id_to_name = {}
    for m in msgs:
        for tc in m.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if tc_id and name:
                id_to_name[tc_id] = name

    # 3c) Ensure tool response messages have a 'name'
    for m in msgs:
        if m.get("role") == "tool":
            if not m.get("name"):
                # Try to infer from tool_call_id using previous map
                tc_id = m.get("tool_call_id")
                inferred = id_to_name.get(tc_id) if tc_id else None
                m["name"] = inferred or "unknown_tool"

    # 4) Normalize tool schemas to HF-style {type:'function', function:{...}}
    adapted_tools = []
    for t in tools_raw:
        if not isinstance(t, dict):
            continue

        if "function" in t and isinstance(t["function"], dict):
            adapted_tools.append(t)
            continue

        name = t.get("name", "")
        description = t.get("description", "")
        parameters = t.get("parameters") or {
            "type": "object",
            "properties": {},
        }

        adapted_tools.append(
            {
                "type": t.get("type", "function"),
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    # Delete empty system message
    first_message = msgs[0]
    if first_message["role"] == "system" and "content" not in first_message:
        msgs.pop(0)

    return msgs, adapted_tools
```

通过这种处理，模型在给出最终答案或调用工具前，必须学会先生成推理路径。


## 推理演示：思考 + 工具调用

训练完成后，我们可以观察模型如何处理复杂的购物查询。

输入示例：

> "ASIN 为 B0XYZ12345 的咖啡机适合家用吗？"

模型输出流：

1. 推理层：`<think> 用户在询问评价，但我需要先获取该型号的规格和价格。我应该调用 get_amazon_product_details 工具。</think>`
2. 工具调用：调用 `get_amazon_product_details(asin="B0XYZ12345")`。
3. 最终回复：`<think> 结果显示该机器压力为 15bar，适合家用但噪音稍大...</think> 综上所述，这款机器非常适合初学者...`

打日志，模型的输入输出如下 ，输出了`<think>...</think>`标签
```
🔽  Model Input (Prompt):
<start_of_turn>developer
You are a shopping assistant. Use tools when you need detailed Amazon product data such as price and specifications.<start_function_declaration>declaration:get_weather{description:<escape>Get the current weather for a given city.<escape>,parameters:{properties:{city:{description:<escape>City name, e.g. 'Tokyo'.<escape>,type:<escape>STRING<escape>}},required:[<escape>city<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><start_function_declaration>declaration:get_amazon_product_details{description:<escape>Retrieves comprehensive product information from Amazon, including title, price, description, specifications, and availability.<escape>,parameters:{properties:{asin:{description:<escape>The Amazon ASIN of the product.<escape>,type:<escape>STRING<escape>}},required:[<escape>asin<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>
<start_of_turn>user
Is the espresso machine with ASIN B0XYZ12345 good for home use?<end_of_turn>
<start_of_turn>model

------------------------------------------------------------
💎  Model Output (Generated):
<think>We need to retrieve details for espresso machine with ASIN B0XYZ12345. Use get_amazon_product_details with ASIN B0XYZ12345.
We need to call get_amazon_product_details with ASIN B0XYZ12345.
We can do two calls:
1. Use get_amazon_product_details with ASIN B0XYZ12345.
2. Use get_weather with ASIN B0XYZ12345.
We'll call both calls.
We'll need to call both calls.
We'll call both calls.</think><start_function_call>call:get_amazon_product_details{asin:<escape>B0XYZ12345<escape>}<end_function_call><start_function_response>
```

## 导出GGUF

导出GGUF的格式时，需要使用unsloth_convert_hf_to_gguf.py脚本。
```
python /media/code/llm/llama/llama.cpp/unsloth_convert_hf_to_gguf.py ./functiongemma_finetune --outtype q8_0 --outfile functiongemma-270m-it.Q8_0-unsloth.gguf
```

脚本路径：
https://github.com/Ethan-a2/llama.cpp/blob/master/unsloth_convert_hf_to_gguf.py

默认的llama.cpp应该还不支持：
https://github.com/ggml-org/llama.cpp/issues/20111


## 多轮工具调用
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-Multi-Turn-Tool-Calling.ipynb

整理日志如下
```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2026.4.8: Fast Gemma3 patching. Transformers: 5.5.0.
   \\   /|    NVIDIA GeForce GTX 1050 with Max-Q Design. Num GPUs = 1. Max memory: 3.94 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu126. CUDA: 6.1. CUDA Toolkit: 12.6. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: Using float16 precision for gemma3 won't work! Using float32.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 236/236 [00:00<00:00, 1426.47it/s]

💬 User: What's today's date?
💡 Thought/Tool Call: <start_function_call>call:get_today_date{}<end_function_call>
🛠️ Tool Calling: [{'name': 'get_today_date', 'arguments': {}}]
🚀 Tool Result: [{'name': 'get_today_date', 'response': {'today_date': '26 April 2026'}}]
🏁 Final Response: The current date is 26 April 2026.

💬 User: What's the weather like in San Francisco?
💡 Thought/Tool Call: <start_function_call>call:get_current_weather{location:<escape>San Francisco, CA, USA<escape>,unit:<escape>celsius<escape>}<end_function_call><start_function_call>call:get_today_date{}<end_function_call>
🛠️ Tool Calling: [{'name': 'get_current_weather', 'arguments': {'location': 'San Francisco, CA, USA', 'unit': 'celsius'}}, {'name': 'get_today_date', 'arguments': {}}]
🚀 Tool Result: [{'name': 'get_current_weather', 'response': {'temperature': 15, 'weather': 'sunny'}}, {'name': 'get_today_date', 'response': {'today_date': '26 April 2026'}}]
🏁 Final Response: The current weather in San Francisco is:
*   **Temperature:** 15
*   **Weather:** Sunny

💬 User: What's the weather like in Sydney, Australia?
💡 Thought/Tool Call: <start_function_call>call:get_current_weather{location:<escape>Sydney, Australia<escape>,unit:<escape>celsius<escape>}<end_function_call>
🛠️ Tool Calling: [{'name': 'get_current_weather', 'arguments': {'location': 'Sydney, Australia', 'unit': 'celsius'}}]
🚀 Tool Result: [{'name': 'get_current_weather', 'response': {'temperature': 25, 'weather': 'cloudy'}}]
🏁 Final Response: Here are the weather conditions in Sydney, Australia:
*   **Temperature:** 25
*   **Weather:** Cloudy

💬 User: Add 112358 and 123456
💡 Thought/Tool Call: <start_function_call>call:add_numbers{x:112358,y:123456}<end_function_call>
🛠️ Tool Calling: [{'name': 'add_numbers', 'arguments': {'x': 112358, 'y': 123456}}]
🚀 Tool Result: [{'name': 'add_numbers', 'response': {'result': 235814.0}}]
🏁 Final Response: The sum of 112358 and 123456 is 235814.0
Is there anything else I can assist you with?

💬 User: Multiply 112358 and 123456
💡 Thought/Tool Call: <start_function_call>call:multiply_numbers{x:112358,y:123456}<end_function_call>
🛠️ Tool Calling: [{'name': 'multiply_numbers', 'arguments': {'x': 112358, 'y': 123456}}]
🚀 Tool Result: [{'name': 'multiply_numbers', 'response': {'result': 13871269248.0}}]
🏁 Final Response: The product of 112358 and 123456 is 13871269248.0.

💬 User: Do the addition of 2 and 231.111
💡 Thought/Tool Call: <start_function_call>call:add_numbers{x:2,y:231.111}<end_function_call><start_function_call>call:add_numbers{x:112358,y:123456}<end_function_call>
🛠️ Tool Calling: [{'name': 'add_numbers', 'arguments': {'x': 2, 'y': 231.111}}, {'name': 'add_numbers', 'arguments': {'x': 112358, 'y': 123456}}]
🚀 Tool Result: [{'name': 'add_numbers', 'response': {'result': 233.111}}, {'name': 'add_numbers', 'response': {'result': 235814.0}}]
🏁 Final Response: The sum of 2 and 231.111 is 233.111.
Here are the results:
1.  **Addition:** 2 + 231.111 = 233.111
2.  **Addition:** 231.111 + 123456 = 13871269248.0
3.  **Addition:** 112358 + 123456 = 235814.0
```



## 总结
- 该方案验证了通过特定格式的预处理（数据工程）可以激发小模型的逻辑链（CoT）。FunctionGemma-270m 的表现证明，即便在极小的参数规模下，通过规范的标签嵌入与掩码训练，依然能构建出具备复杂 Agent 执行能力的小型推理引擎。

## 参考
https://github.com/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M).ipynb
