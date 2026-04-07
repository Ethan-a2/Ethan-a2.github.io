# llama.cpp 性能实测：Flash Attention、GQA 与 YaRN 长上下文的提速魔法

在本地部署大语言模型（LLM）时，**推理速度**和**上下文长度**是我们最关心的核心指标。我们基于 `llama.cpp` (CUDA 后端) 进行了一系列深度的基准测试，覆盖了 MHA（多头注意力）与 GQA（分组查询注意力）的对比、Flash Attention 的加速效果，以及 YaRN 机制在处理 32K 超长上下文时的表现。

本文将为您拆解这份详实的硬核测试报告。

---

## 1. MHA 与 GQA 的基础对比：LLaMA-2 vs LLaMA-3

首先，我们对比了采用标准 MHA 的 LLaMA-2 (7B) 和采用 GQA 架构的 LLaMA-3.2 (8B)。

**测试环境：**
- **测试提示词**：长上下文测试提示，重复 10 次
- **参数配置**：`-n 100 --temp 0 --n-predict 100`
- **量化格式**：`Q4_K_M`
    
|**模型 (架构)**|**Prompt 处理速度**|**生成速度**|
|---|---|---|
|**LLaMA-2 7B** (MHA)|42.6 t/s|7.1 t/s|
|**LLaMA-3.2 8B** (GQA)|42.5 t/s|6.6 t/s|

**💡 观察结论：**

在此特定测试条件下，尽管 LLaMA-3.2 的参数量（8B）略大于 LLaMA-2（7B），但得益于 GQA（`n_kv_head=8`, `n_head=32`）的架构优化，其 Prompt 处理速度与 LLaMA-2 几乎持平（42.5 t/s vs 42.6 t/s），生成速度也仅有微小的差距。GQA 在显存占用和计算效率上的优势，使得更大规模的模型依然能保持优秀的吞吐量。


验证方法：
```
PROMPT="长上下文测试提示，重复 10 次以增加长度"

MHA:
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli \
  -m llama-2-7b.Q4_K_M.gguf \
  -p "$PROMPT" \
  -n 100 --temp 0 \
  --n-predict 100



GQA（LLaMA-3 8B, n_kv_head=8, n_head=32）:
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli \
  -m Llama-3.2-8B-Instruct.Q4_K_M.gguf \
  -p "$PROMPT" \
  -n 100 --temp 0 \
  --n-predict 100
```

---

## 2. Flash Attention：性能起飞的“物理外挂”

Flash Attention (FA) 是近年来 LLM 推理优化的重头戏。我们使用 **LLaMA-3.2-1B-Instruct** (`Q4_K_M`) 进行了开启与关闭 FA 的极速对比。

### 命令行快速直测 (`-c 8192`)

通过简单的命令行参数 `--flash-attn 0` 与 `--flash-attn 1`，我们可以看到惊人的差距：

|**模式**|**Prompt 处理速度 (Prefill)**|**文本生成速度 (Decode)**|
|---|---|---|
|**标准 Attention**|335.6 t/s|45.9 t/s|
|**Flash Attention**|**734.1 t/s** (🚀 **提升 118%**)|**49.4 t/s** (🚀 **提升 7.6%**)|

- 验证方法:
```
export LD_LIBRARY_PATH=./llama.cpp/pkg-cuda/llama.cpp/lib:$LD_LIBRARY_PATH

# 标准 Attention
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "长上下文测试提示，重复 10 次以增加长度" \
  -n 256 -c 8192 --flash-attn 0 -ngl 99 --single-turn

# Flash Attention（对比）
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "长上下文测试提示，重复 10 次以增加长度" \
  -n 256 -c 8192 --flash-attn 1 -ngl 99 --single-turn
```

### Llama-Bench 基准测试

为了获得更严谨的数据，我们使用了内置的 `llama-bench` 工具进行跑分验证：

|**测试项目**|**标准 Attention (t/s)**|**Flash Attention (t/s)**|**性能提升幅度**|
|---|---|---|---|
|**Prompt 处理 (pp2048)**|43.73 ± 0.05|**54.45 ± 0.07**|**+24.5%**|
|**文本生成 (tg128)**|51.84 ± 0.16|**55.52 ± 0.05**|**+7.1%**|

**💡 观察结论：**

开启 Flash Attention 是一个**无需思考的必选项**。它对 Prompt 预填充阶段（Prefill）有着翻倍级别的提速效果，尤其在处理大段前置提示词时体验极佳，同时对生成阶段也有小幅度的增益。

```
# 标准 Attention
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-bench -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 2048 -n 128 -fa 0 -ngl 99 -b 1 -r 2

# Flash Attention
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-bench -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 2048 -n 128 -fa 1 -ngl 99 -b 1 -r 2
```
---

## 3. 质量验证：Flash Attention 是“免费的午餐”吗？

性能提升了，生成质量会下降吗？我们使用 `llama-perplexity` 工具在 `wikitext-2` 数据集上进行了困惑度（PPL）测试。

> _注：PPL 值越低，代表模型对文本的预测能力越强，输出质量越高。_

- **关闭 FA (fa 0):** PPL = 15.7593 +/- 0.61082
    
- **开启 FA (fa 1):** PPL = 15.7571 +/- 0.61098
    

**💡 观察结论：**

两者差距在小数点后第三位，完全处于误差范围内。这证明了 **Flash Attention 带来的性能翻倍是完全“无损”的**。


```
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-perplexity -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -f wikitext-2-raw/wiki.test.small.raw -fa 0

./llama.cpp/pkg-cuda/llama.cpp/bin/llama-perplexity -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -f wikitext-2-raw/wiki.test.small.raw -fa 1
```
---

## 4. 挑战 32K 极限：标准 RoPE vs YaRN 扩展

在进行“大海捞针”（Needle-in-a-Haystack）测试时，我们要求模型处理长达 32768 tokens 的文本 (`needle_32k.txt`)。如果直接使用默认配置，部分模型极易在超出其原生训练长度后崩溃或出现严重幻觉。

我们对比了标准 RoPE 与引入 YaRN 扩展机制（支持 4 倍上下文扩展）的表现：

|**配置策略**|**执行参数摘录**|**Prompt 速度**|**生成速度**|
|---|---|---|---|
|**标准 RoPE**|`-c 32768`|293.7 t/s|31.5 t/s|
|**YaRN 扩展**|`--rope-scaling yarn --rope-scale 4`|294.5 t/s|31.7 t/s|

**💡 观察结论：**

引入 YaRN 机制（`--rope-scaling yarn` 等参数）进行 4x 上下文扩展时，**几乎没有任何性能损耗**。无论 Prompt 处理还是文本生成，速度表现都异常坚挺。这意味着你可以放心开启 YaRN 来解锁模型的长文本阅读能力，而不必担心拖慢推理速度。

验证方法:
```
# 标准 RoPE（容易崩溃在训练长度外）
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli \
  -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -f long_prompt.txt \
  -n 100 --temp 0 -c 32768

# YaRN 扩展（支持 4x 上下文）
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli \
  -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -f long_prompt.txt \
  -n 100 --temp 0 -c 32768 \
  --rope-scaling yarn \
  --rope-scale 4 \
  --yarn-ext-factor 1.0
```
---

## 总结建议

基于以上测试报告，对于使用 `llama.cpp` 进行本地部署的开发者，我们强烈建议：
1. **永远开启 Flash Attention** (`-fa 1`)：在不损失任何生成质量的前提下，换取巨幅的预处理提速。
2. **善用 YaRN 处理长文本**：如果你需要处理超越模型原生支持长度的代码库或文档，使用 `--rope-scaling yarn` 是一种高效且对性能极度友好的扩展方案。
3. **拥抱 GQA 架构模型**：类似 LLaMA-3 这样采用 GQA 的新一代模型，在参数量增加的情况下依然能保持极高的推理效率，值得优先考虑。
