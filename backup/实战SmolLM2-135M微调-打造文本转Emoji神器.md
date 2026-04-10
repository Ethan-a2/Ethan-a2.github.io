在大型语言模型（LLM）动辄百亿参数的今天，轻量级模型（Small Language Models, SLMs）正变得越来越受欢迎。它们不仅能在消费级显卡甚至 CPU 上流畅运行，而且在特定任务上的表现往往能带来惊喜。

今天，我们将以 **SmolLM2-135M-Instruct** 为基础，手把手带你完成一次极其有趣的微调任务：**教模型把文字翻译成纯 Emoji 表情！** 本教程不仅包含完整的代码，还会详细解释从数据准备到 LoRA 权重合并的全过程，特别是从其他模型（如 Gemma）迁移到 SmolLM2 时需要注意的“坑”。

---

## 1. 为什么选择 SmolLM2-135M？

SmolLM2 是 Hugging Face 推出的一系列极具竞争力的轻量级模型。135M（1.35亿参数）的版本非常小巧，加载速度极快，且 Instruct 版本已经经过了良好的指令微调，原生支持标准的 System/User/Assistant 对话模板。这使得它非常适合用来做单点任务（如格式转换、情感分析、翻译）的定制化微调。

---

## 2. 环境与硬件准备

首先，我们需要检测当前的硬件环境。如果你有 GPU，训练速度会起飞；如果没有，135M 的体量也允许你在 CPU 上跑完实验。

```python
import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
DTYPE = torch.bfloat16 if USE_CUDA else torch.float32
print(f"Detected device: {DEVICE}, dtype: {DTYPE}, CUDA available: {USE_CUDA}")
```

> **Tip:** 如果你的显卡较新（如 RTX 30/40 系列），强烈建议使用 `bfloat16`，它能在保持精度的同时大幅节省显存并加速训练。

---

## 3. 数据集：筛选最纯正的 Emoji

我们使用的是 `kr15t3n/text2emoji` 数据集。为了防止模型在输出时夹杂奇怪的文字，我们需要对数据进行严格过滤，确保 `emoji` 字段里**只包含 Emoji 符号**。

```python
import emoji
from datasets import load_dataset

general_dataset_path = load_dataset("kr15t3n/text2emoji", encoding="utf-8", split="train")

def is_only_emoji(sample):
    emoji_string = sample["emoji"]
    if not emoji_string:
        return False
    return emoji.purely_emoji(emoji_string)

dataset = general_dataset_path.filter(is_only_emoji)
```

随后，我们要把数据格式化为 SmolLM2-Instruct 认识的“聊天记录”形式：

```python
def translate(sample):
    return {
        "messages": [
            {"role": "system",    "content": "Translate this text to emoji."},
            {"role": "user",      "content": sample["text"]},
            {"role": "assistant", "content": sample["emoji"]},
        ]
    }

training_dataset = dataset.map(translate, remove_columns=dataset.features.keys())
training_dataset_splits = training_dataset.train_test_split(test_size=0.1, shuffle=True)
```
加入 `System Prompt` 非常关键，这等于在给模型下达一个全局的指令。

---

## 4. 加载模型与避坑指南

加载 SmolLM2 时，有一个非常重要的细节需要注意：**Pad Token 的设置**。

SmolLM2 默认的 `pad_token` 是 `None`。如果在微调时不显式设置它，模型在处理不等长的数据批次（Batch）时就会报错，或者导致训练 Loss 崩溃。我们的解决方案是将其与 `eos_token` 绑定。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

smol_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(smol_model)

# 【关键避坑】为 SmolLM2 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    smol_model,
    device_map="auto" if USE_CUDA else "cpu",
    torch_dtype=DTYPE,
)
base_model.config.pad_token_id = tokenizer.pad_token_id
```

---

## 5. 配置 QLoRA：花小钱办大事

为了进一步降低显存占用，我们采用 **QLoRA**（4-bit 量化 + LoRA）技术。

在使用 LoRA 时，我们需要指定哪些网络层需要被微调（`target_modules`）。不同模型的线性层命名各不相同（如 `q_proj`, `k_proj`, `gate_proj` 等）。为了省去查阅架构文档的麻烦，我们可以直接使用 **`"all-linear"`** 这个魔法参数，让系统自动覆盖所有线性层。

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig

# 1. 4-bit 量化配置
if USE_CUDA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

# 2. LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear", # 自动捕获所有线性层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

在训练参数方面，由于 135M 模型很小，我们可以适当地将 `per_device_train_batch_size` 调大到 `2` 或 `4`，以加快训练速度。

---

## 6. 开启 SFT（监督微调）

借助 Hugging Face 的 `trl`（Transformer Reinforcement Learning）库，微调只需几行代码。我们使用 `SFTTrainer` 将模型、数据和 LoRA 配置结合起来。

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=base_model,
    args=args, # 包含 batch size, learning rate (5e-5) 等的 SFTConfig
    train_dataset=training_dataset_splits["train"],
    eval_dataset=training_dataset_splits["test"],
    peft_config=lora_config,
)
trainer.train()

# 保存 LoRA 适配器权重
adapter_path = "./content/myemoji-smollm2-adapters"
trainer.save_model(adapter_path)
```
在训练过程中，你可以提取 `trainer.state.log_history` 来绘制 Training Loss 和 Validation Loss 的曲线图，直观地观察模型的收敛情况。

---

## 7. 合并权重：让模型融为一体

微调结束后，我们得到的是独立的 LoRA 适配器（Adapter）权重。为了方便部署和后续使用，最佳实践是将其与基础模型合并（Merge）。

```python
from peft import PeftModel

merged_model_path = "./content/myemoji-smollm2-merged"

# 重新加载基础模型和分词器
base_model = AutoModelForCausalLM.from_pretrained(smol_model, device_map="auto")
tokenizer  = AutoTokenizer.from_pretrained(adapter_path)

# 加载适配器并合并
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

# 保存最终的一体化模型
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
```

---

## 8. 见证奇迹的时刻

最后，让我们写一段推理代码，对比一下 **未经微调的基础模型** 和 **微调后的模型** 在同一句话上的表现：

**输入文本：** `"let's go to the beach"`

* **基础模型输出：** 输出一句啰嗦的话， *"I'm going to the beach, just like we did yesterday. We've got all the amenities we need and the kids are excited to go swimming"* 甚至没有产生 Emoji。
* **微调后输出：** `🏖️🌊☀️`

仅仅使用了极少的资源和不到半小时的时间，我们就把一个通用的问答模型，变成了一个专注的“表情包翻译机”。

## 总结
微调小模型（SLM）的过程不仅成本低廉，更是深入理解 LLM 训练流程的最佳途经。通过合理处理 `pad_token`、利用 `"all-linear"` 自动配置 LoRA、以及规范化 System Prompt，我们可以轻松将 SmolLM2 打造成满足各类细分垂直需求的 AI 助手。


## 训练日志

<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/80b4f40f-2bf1-4dc9-a823-5a39df5314a0" />

```
python ./Fine_tune_SmolLM2-135M_for_emoji_generation.py
Detected device: cuda, dtype: torch.bfloat16, CUDA available: True
Filter: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2529/2529 [00:00<00:00, 37723.10 examples/s]

Here's the 10th example from the dataset: {'text': 'A good stretch first thing in the morning feels amazing.', 'emoji': '🙆☀️😌'}
config.json: 861B [00:00, 2.23MB/s]
tokenizer_config.json: 3.76kB [00:00, 2.30MB/s]
vocab.json: 801kB [00:00, 1.07MB/s]
merges.txt: 466kB [00:00, 600kB/s]
special_tokens_map.json: 655B [00:00, 575kB/s]
tokenizer.json: 2.10MB [00:01, 1.72MB/s]
`torch_dtype` is deprecated! Use `dtype` instead!
model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 269M/269M [00:29<00:00, 9.04MB/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 272/272 [00:00<00:00, 2812.46it/s]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 132/132 [00:00<00:00, 554kB/s]
Device: cuda:0
DType:  torch.bfloat16
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2529/2529 [00:00<00:00, 8400.50 examples/s]

Here's the 40th example from the formatted training dataset:
{'messages': [{'content': 'Translate this text to emoji.', 'role': 'system'}, {'content': 'Afternoon tea with scones', 'role': 'user'}, {'content': '☕️🍰🍓👑', 'role': 'assistant'}]}
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 272/272 [00:00<00:00, 4886.48it/s]
Passing `generation_config` together with generation-related arguments=({'max_new_tokens'}) is deprecated and will be removed in future versions. Please pass either a `generation_config` object OR all generation parameters explicitly, but not both.
Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

Dataset text:          Let them cook
Dataset emoji:         🔥🧑‍🍳
Model generated output:"Enough to throw away the dinner, let it go out and be done. That's enough."
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 272/272 [00:00<00:00, 802.26it/s]
Training configured
Tokenizing train dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2276/2276 [00:00<00:00, 2811.64 examples/s]
Tokenizing eval dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 253/253 [00:00<00:00, 2502.63 examples/s]
{'loss': '1.363', 'grad_norm': '1.094', 'learning_rate': '5e-05', 'entropy': '1.338', 'num_tokens': '9.289e+04', 'mean_token_accuracy': '0.705', 'epoch': '1'}                                              
{'eval_loss': '1.17', 'eval_runtime': '8.043', 'eval_samples_per_second': '31.45', 'eval_steps_per_second': '3.978', 'eval_entropy': '1.22', 'eval_num_tokens': '9.289e+04', 'eval_mean_token_accuracy': '0.7387', 'epoch': '1'}                                                                                                                                                                                        
 37%|██████████████████████████████████████████████████████████████▌                                                                                                         | 1271/3414 [12:38<22:43,  1.57 37%|████████████████████████████████████████████████████████████▋                                                                                                      | 1272/3414 [12:39<23:44,  1.50it/s]{'loss': '1.178', 'grad_norm': '1.109', 'learning_rate': '5e-05', 'entropy': '1.18', 'num_tokens': '1.858e+05', 'mean_token_accuracy': '0.7347', 'epoch': '2'}                                              
{'eval_loss': '1.137', 'eval_runtime': '8.652', 'eval_samples_per_second': '29.24', 'eval_steps_per_second': '3.699', 'eval_entropy': '1.177', 'eval_num_tokens': '1.858e+05', 'eval_mean_token_accuracy': '0.7406', 'epoch': '2'}                                                                                                                                                                                      
{'loss': '1.145', 'grad_norm': '1.289', 'learning_rate': '5e-05', 'entropy': '1.15', 'num_tokens': '2.787e+05', 'mean_token_accuracy': '0.7406', 'epoch': '3'}                                              
{'eval_loss': '1.117', 'eval_runtime': '8.109', 'eval_samples_per_second': '31.2', 'eval_steps_per_second': '3.946', 'eval_entropy': '1.144', 'eval_num_tokens': '2.787e+05', 'eval_mean_token_accuracy': '0.745', 'epoch': '3'}                                                                                                                                                                                        
{'train_runtime': '2279', 'train_samples_per_second': '2.996', 'train_steps_per_second': '1.498', 'train_loss': '1.229', 'epoch': '3'}                                                                      
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3414/3414 [37:59<00:00,  1.50it/s]
LoRA adapters saved to ./content/myemoji-smollm2-adapters
/home/liuqi/.local/lib/python3.10/site-packages/matplotlib/projections/__init__.py:63: UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed (e.g. as a system package and as a pip package). As a result, the 3D projection is not available.
  warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 272/272 [00:00<00:00, 846.13it/s]
Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.90it/s]
Merged model saved to ./content/myemoji-smollm2-merged
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 272/272 [00:00<00:00, 1920.67it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 272/272 [00:00<00:00, 2842.43it/s]
Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

Fine-tuned model: 🏖️🌊☀️
Base model:       I'm going to the beach, just like we did yesterday. We've got all the amenities we need and the kids are excited to go swimming.
```
