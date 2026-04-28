
在端侧或显存受限的设备上运行大模型时，随着上下文长度（Context Length）的增加，KV Cache 占用的显存会呈线性增长，甚至超过模型权重本身的占用。为了缓解这一问题，`llama.cpp` 引入了 KV Cache 量化功能。

本文将以 Llama-3.2-1B-Instruct 为例，深入对比 `f16`（基线）、`q8_0`、`q4_0` 以及混合量化（`K q8_0 + V q4_0`）在精度损失 (Perplexity)、吞吐性能 (Throughput) 以及 显存占用 (VRAM) 三个维度的实际表现。

## 🧪 测试环境与基础配置

- 模型: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
    
- 推理框架: `llama.cpp` (CUDA backend, Flash Attention 开启)
    
- 测试数据集: `wikitext-2-raw/wiki.test.small.raw`
    
- 硬件监控: `nvidia-smi` 实时采样
    

---

## 1. 精度损失验证 (Perplexity)

我们首先使用 `llama-perplexity` 工具测试了不同 KV Cache 量化策略对模型生成质量的影响。测试上下文长度（CTX）设定为 1024，分 50 个 chunk 进行评估。PPL（困惑度）越低，代表模型输出质量越高、精度损失越小。

结果:

| kvcache        | ppl                 |
| -------------- | ------------------- |
| f16            | 14.1462 +/- 0.52672 |
| q8_0           | 14.1683 +/- 0.52746 |
| q4_0           | 14.7469 +/- 0.54700 |
| K q8_0  V q4_0 | 14.2329 +/- 0.52982 |


💡 分析结论：
- 8-bit 量化极其出色： `q8_0` 的 PPL 表现与 `f16` 几乎完全一致，可以说是“无损”压缩。
- 4-bit 量化存在代价： 直接将 K 和 V 都压缩到 `q4_0` 会导致 PPL 出现肉眼可见的上升（从 14.1 升至 14.7）。
- 混合量化的折中： 采用 `K q8_0` 配合 `V q4_0`，可以把精度损失控制在一个非常理想的范围内，似乎是一个兼顾质量和压缩率的好方案（但请继续看下文的性能测试）。



验证方法:
```
MODEL="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
//MODEL="Qwen3-0.6B-Q4_K_M.gguf"
DATASET="./wikitext-2-raw/wiki.test.small.raw"
CTX=1024

export LD_LIBRARY_PATH=./llama.cpp/pkg-cuda/llama.cpp/lib:$LD_LIBRARY_PATH

# --- baseline: f16 KV ---  这个是基础
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-perplexity \
  -m $MODEL -f $DATASET \
  -c $CTX --chunks 50 \
  --cache-type-k f16 --cache-type-v f16

Final estimate: PPL = 14.1462 +/- 0.52672


# --- q8_0 KV ---
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-perplexity \
  -m $MODEL -f $DATASET \
  -c $CTX --chunks 50 \
  --cache-type-k q8_0 --cache-type-v q8_0

Final estimate: PPL = 14.1683 +/- 0.52746



# --- q4_0 KV ---
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-perplexity \
  -m $MODEL -f $DATASET \
  -c $CTX --chunks 50 \
  --cache-type-k q4_0 --cache-type-v q4_0

Final estimate: PPL = 14.7469 +/- 0.54700


# --- K q8_0  V q4_0 ---
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-perplexity \
  -m $MODEL -f $DATASET \
  -c $CTX --chunks 50 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --flash-attn 1 

Final estimate: PPL = 14.2329 +/- 0.52982

```

---

## 2. 吞吐量与延迟测试 (Throughput & Latency)

精度只是一方面，推理速度同样关键。我们使用 `llama-bench` 工具，设定 Prompt 长度 512，生成长度 128，开启 Flash Attention (`--flash-attn 1`)，测试了不同策略的吞吐量（t/s）。


💡 分析结论（重点避坑）：
- 同构量化性能损耗极小： 无论是纯 `q8_0` 还是纯 `q4_0`，其 Prompt 处理速度和生成速度仅比 `f16` 基线下降了 1% - 3%，在实际体验中完全无感。
- 混合量化存在严重性能瓶颈： 令人意外的是，虽然 `K q8_0 + V q4_0` 精度不错，但它的 Prompt 处理速度出现了断崖式下跌（从 1076 t/s 暴跌至 213 t/s，下降了近 80%），生成速度也有所下降。
- _原因剖析：_ 这通常是因为当前 CUDA 后端针对“异构”的 KV Cache 类型缺乏高度优化的 Flash Attention Kernel，导致在计算 Attention 时产生了巨大的额外开销。
    
 - 验证方法及结果：
 ```
 ./llama.cpp/pkg-cuda/llama.cpp/bin/llama-bench \
  -m $MODEL \
  -p 512 -n 128 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn 1 \
  -r 3

| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CUDA       |  99 |  1 |           pp512 |      1076.18 ± 10.21 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CUDA       |  99 |  1 |           tg128 |         55.84 ± 0.17 |



./llama.cpp/pkg-cuda/llama.cpp/bin/llama-bench \
  -m $MODEL \
  -p 512 -n 128 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn 1 \
  -r 3

| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 |           pp512 |       1059.13 ± 6.62 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 |           tg128 |         53.79 ± 0.16 |




./llama.cpp/pkg-cuda/llama.cpp/bin/llama-bench \
  -m $MODEL \
  -p 512 -n 128 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --flash-attn 1 \
  -r 3

| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 |           pp512 |       1058.30 ± 7.17 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 |           tg128 |         53.69 ± 0.06 |



./llama.cpp/pkg-cuda/llama.cpp/bin/llama-bench \
  -m $MODEL \
  -p 512 -n 128 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --flash-attn 1 \
  -r 3

| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 |           pp512 |       213.91 ± 11.31 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CUDA       |  99 |   q8_0 |   q4_0 |  1 |           tg128 |         47.65 ± 0.15 |

 ```

---

## 3. 显存节省验证 (VRAM Optimization)

量化的最终目的是省显存。为了放大差异，我们将上下文长度拉长到 16384 (16K)，使用 `llama-cli` 运行单轮长文本生成（Prompt: "请写一篇5000字的文章"），并通过 `nvidia-smi` 记录显存峰值。


| kvcache        | VRAM   |
| -------------- | ------ |
| f16            | 1580MB |
| q8_0           | 1354MB |
| q4_0           | 1226MB |
| K q8_0  V q4_0 | 1286MB |



💡 分析结论：
- 在 16K 长上下文中，`q8_0` 能稳定节约超过 200MB 的显存，而极致的 `q4_0` 更是能省下 350MB 以上的显存空间。对于显存卡在临界点的用户（比如 8G 显存跑 8B 模型长文本），这就是跑得通和 OOM 的区别。
    

- 验证方法
```
# 终端1：运行推理
./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli -m $MODEL -c 16384 \
  --cache-type-k f16 --cache-type-v f16 \
  -p "请写一篇5000字的文章" -n 1000 --single-turn

1580MB


./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli -m $MODEL -c 16384 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -p "请写一篇5000字的文章" -n 1000 --single-turn

1354MB



./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli -m $MODEL -c 16384 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -p "请写一篇5000字的文章" -n 1000 --single-turn

1226MB



./llama.cpp/pkg-cuda/llama.cpp/bin/llama-cli -m $MODEL -c 16384 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  -p "请写一篇5000字的文章" -n 1000 --single-turn --flash-attn 1

1286MB



# 终端2：采样显存
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

---

## 最终总结与配置建议

综合上述测试结果，我们对 `llama.cpp` 的 KV Cache 量化策略得出以下实战建议：

1. 🏆 首选推荐：纯 `q8_0` 量化 (`--cache-type-k q8_0 --cache-type-v q8_0`)
    - 理由： 精度几乎零损失，推理速度与 `f16` 几乎持平，同时能有效降低长上下文的显存占用。性价比极高。
2. 极限显存拯救者：纯 `q4_0` 量化 (`--cache-type-k q4_0 --cache-type-v q4_0`)
    - 理由： 显存优化效果最明显。如果你的应用场景对微小的逻辑/语法错误容忍度较高，或者显存实在捉襟见肘，可以使用该方案。速度依然有保障。
3. ❌ 绝对避坑：混合量化 (`--cache-type-k q8_0 --cache-type-v q4_0`)
    - 理由： 尽管在纸面精度和显存占用上表现平衡，但在目前的底层算子实现中，会导致极其严重的性能下降。在 llama.cpp 针对异构 KV Cache 优化内核之前，强烈建议不要使用这种组合。
