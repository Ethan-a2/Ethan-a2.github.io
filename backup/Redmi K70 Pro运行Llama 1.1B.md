
# 前言

随着移动设备硬件性能的飞速发展，将大型语言模型（LLM）部署到端侧设备上运行，实现低延迟、保护用户隐私的AI推理，正成为一个热门方向。本文记录了在搭载高通骁龙8 Gen3处理器的Redmi K70 Pro上，尝试运行其提供的Llama 1.1B MobileQuant模型的过程、遇到的问题及初步分析。

# 环境准备

*   **设备型号：** Redmi K70 Pro
*   **处理器：** 高通骁龙8 Gen3 (Snapdragon 8 Gen3)
*   **操作系统：** Android（通过ADB连接和操作）
*   **工具：** ADB (Android Debug Bridge)
*   **模型与推理程序来源：** Hugging Face (fwtan/llm_8gen3_demo, fwtan/llama-1.1b-mobilequant-w4a8-s1024-e60-sym-8gen3)

# 模型与程序下载

首先，我们需要从Hugging Face下载MobileQuant为骁龙8 Gen3准备的推理Demo程序和Llama 1.1B的量化模型。

```bash
# 下载推理程序 demo
huggingface-cli download fwtan/llm_8gen3_demo --local-dir .

# 下载 Llama 1.1B 量化模型
# llama-1.1b-mobilequant-w4a8-s1024-e60-sym-8gen3
# w4a8: 4-bit 权重，8-bit 激活
# s1024: sequence length 1024
# e60: 可能是 epoch 60
# sym: symmetric quantization
huggingface-cli download fwtan/llama-1.1b-mobilequant-w4a8-s1024-e60-sym-8gen3 --local-dir .
```

下载完成后，您会在当前目录下看到两个文件/文件夹：`llm_8gen3_demo` (一个可执行文件) 和 `llama-1.1b-mobilequant-w4a8-s1024-e60-sym-8gen3` (模型文件)。

# 文件部署到设备

使用ADB将下载的文件推送到Redmi K70 Pro上。为了方便，我们统一存放在`/data/local/tmp/`目录下。

```bash
adb push llm_8gen3_demo /data/local/tmp/

# 将模型文件推送到设备，放在demo程序同级目录下
adb push llama-1.1b-mobilequant-w4a8-s1024-e60-sym-8gen3 /data/local/tmp/llm_8gen3_demo
```


# 手动在设备上运行

通过ADB shell进入设备终端，执行程序。

```bash
# 进入部署目录
adb shell
cd /data/local/tmp/llm_8gen3_demo

# 设置库路径，使得可执行文件能够找到其依赖的共享库和DSP库
export LD_LIBRARY_PATH=$PWD
export ADSP_LIBRARY_PATH=$PWD

# 赋予 simple_app 可执行权限
# 注意：可执行文件的名字是 simple_app
chmod +x simple_app

# 运行推理程序，并指定模型文件目录
./simple_app llama-1.1b-mobilequant-w4a8-s1024-e60-sym-8gen3
```

# 实测结果

程序成功启动，并提示输入。我们尝试输入一个简单的指令："tell me a joke."

```
Hello, how can I help you today?
>>> tell me a joke.
ink is a liquid that is used to write and draw. It is made from the same ingredients as ink, but it is thicker and more viscous than traditional ink. The ink is made by mixing a dye with a pigment, which is then mixed with a binder and a solvent. The binder is a mixture of water and alcohol, which helps to bind the pigment to the paper. The solvent is a chemical that helps to remove any excess pigment from the paper. The ink is then coated with a layer of wax to help it stick to the paper. The result is a water-resistant ink that is easy to write and draw on.</s>(19.784995 tok/s)
>>> 
```

<img width="1207" height="140" alt="Image" src="https://github.com/user-attachments/assets/ee3cfe3c-0c34-4cfd-ac0b-e996d79b1173" />

**观察结果：答非所问**

从输出可见，模型并没有给出与“笑话”相关的内容，而是输出了一段关于“墨水”的描述。这明显是**答非所问**的现象。

# 结果分析与问题探讨

尽管我们成功在Redmi K70 Pro上运行了MobileQuant以及Llama 1.1B模型，但推理的质量未能达到预期。导致“答非所问”的原因可能由以下几点综合影响：

1.  **模型规模与量化程度：**
    *   **Llama 1.1B 是一个非常小的模型。** 即使是未量化的原始Llama 1.1B，其生成能力也相对有限，很难处理复杂的指令或生成高质量的创意内容（如笑话）。它更多是作为大规模模型的基座或用于极其简单的任务。
    *   **W4A8 (4-bit 权重，8-bit 激活) 是非常激进的量化方案。** 深度量化虽然能大幅减小模型体积、提升运行速度，但也会带来信息损失和精度下降，尤其对小模型的影响更为显著。这可能导致模型语义理解能力和生成连贯性严重受损，产生“幻觉”或完全不相关的输出。模型可能只是在拼凑词语，而无法理解其深层含义或指令意图。

2.  **模型是否为指令微调版（Instruction-tuned）：** 许多公开的Llama模型是基础模型，需要经过指令微调（Instruction-tuning）才能更好地理解并遵循用户指令。如果这个Llama 1.1B模型是未经指令微调的基座模型，那么即使是高质量的输入，也可能无法得到预期的交互式回答。它可能只是在按其预训练数据中的概率生成文本。

3.  **Demo应用程序的局限性：** 示例程序`simple_app`可能只是一个非常基础的推理封装，它可能没有包含任何形式的Prompt Engineering（例如，系统提示词、多轮对话管理、CoT/CoH等策略）。它可能只是简单地将用户输入拼接后进行推理，这对于需要特定格式或上下文才能良好工作的模型来说是不够的。

4.  **模型与运行时的兼容性或 bug：** 虽然可能性相对较低，但也不能完全排除。在特定硬件或运行时版本上，模型加载或推理计算过程中可能存在导致数据损坏或行为异常的bug。

# 结论与展望

本次测试成功验证了在Redmi K70 Pro这样的高端移动设备上运行MobileQuant推理框架的可行性。设备能够加载并执行模型，并给出了接近20 tok/s的推理速度，这对于端侧LLM来说是一个不错的开始。

然而，Llama 1.1B W4A8模型的实际推理质量表现不佳，无法用于实际的问答场景。这提示我们：

*   **模型质量至关重要：** 即使运行速度快，如果模型本身的质量（包括其原始能力和量化后的精度）不足，也无法提供有用的服务。
*   **端侧微调与更优量化：** 未来尝试部署到端侧的模型，可能需要选择更大、经过更充分指令微调的模型，并结合更先进的非对称量化或混合精度量化方案，以在性能和精度之间取得更好的平衡。
*   **Prompt工程：** 即使是端侧模型，也需要合适的Prompt Engineering来引导其行为，使其更好地理解用户意图。

未来可以尝试该框架支持的更大规模模型或不同量化策略的模型，以探索在移动设备上实现实用级LLM推理的可能性。

# 参考资料
- [MobileQuant/capp at main · saic-fi/MobileQuant](https://github.com/saic-fi/MobileQuant/tree/main/capp)
