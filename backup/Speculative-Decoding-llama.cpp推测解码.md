

- 生成 200 个 token（参数 -n 200），提示为：“重复这个句子10次：这是一个测试提示。”
- 随机种子：--seed 23（保证结果可复现）。

# 基线
```
./llama.cpp/build/bin/llama-cli -m ./LFM2-1.2B-Q4_K_M.gguf -n 200 -p "重复这个句子10次：这是一个测试提示。" --seed 23
```

```
[ Prompt: 196.3 t/s | Generation: 47.6 t/s ]

llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - Host               |                 2580 =   685 +    1500 +     395                |
llama_memory_breakdown_print: |   - CPU_REPACK         |                  482 =   482 +       0 +       0                |
```
- 生成速度：47.6 tokens/秒（t/s）

# Speculative-Decoding
```
./llama.cpp/build/bin/llama-speculative --model-draft ./LFM2-350M-Q4_K_M.gguf -m ./LFM2-1.2B-Q4_K_M.gguf --draft-n 4 -n 200 -p "重复这个句子10次：这是一个测试提示。" --seed 23
```

关键输出：
```
encoded   15 tokens in    0.140 seconds, speed:  107.248 t/s
decoded  202 tokens in    2.533 seconds, speed:   79.745 t/s

n_draft   = 4
n_predict = 202
n_drafted = 376
n_accept  = 107
accept    = 28.457%

draft:

llama_perf_context_print:        load time =     728.24 ms
llama_perf_context_print: prompt eval time =     519.43 ms /    56 tokens (    9.28 ms per token,   107.81 tokens per second)
llama_perf_context_print:        eval time =    1991.26 ms /   159 runs   (   12.52 ms per token,    79.85 tokens per second)
llama_perf_context_print:       total time =    2674.12 ms /   215 tokens
llama_perf_context_print:    graphs reused =        201

target:

common_perf_print:    sampling time =      18.72 ms
common_perf_print:    samplers time =       7.51 ms /   202 tokens
common_perf_print:        load time =     969.35 ms
common_perf_print: prompt eval time =     785.34 ms /   100 tokens (    7.85 ms per token,   127.33 tokens per second)
common_perf_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
common_perf_print:       total time =    3471.78 ms /   101 tokens
common_perf_print: unaccounted time =    2667.72 ms /  76.8 %      (total - sampling - prompt eval - eval) / (total)
common_perf_print:    graphs reused =         16
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - Host               |                 2580 =   685 +    1500 +     395                |
llama_memory_breakdown_print: |   - CPU_REPACK         |                  482 =   482 +       0 +       0                |

```


- 接受率:accept    = 28.457%
- 79.745 t/s，加速比为： 79.745 / 47.6 = 1.68 x


# 对比分析

| 指标                    | 基线模式 (`llama-cli`)                | Speculative Decoding 模式 (`llama-speculative`) | 分析与备注                                           |
| --------------------- | --------------------------------- | --------------------------------------------- | ----------------------------------------------- |
| **总生成时间**             | ~4.2 秒 (200 tokens / 47.6 t/s 估算) | **2.533 秒** (decoded 202 tokens)              | **实际快 46%**，这是最关键的收益。                           |
| **生成速度 (t/s)**        | **47.6** (仅生成阶段)                  | **79.745** (解码总速度)                            | **+68%**                                        |
| **接受率**               | N/A                               | **28.457%** (107 / 376)                       | 目标模型接受的 draft token 比例。此值较低，但仍有显著加速，得益于极小的草稿模型。 |
| **Draft 模型**          | 无                                 | LFM2-350M (Q4_K_M)                            | 用于快速生成草稿，是加速的关键。                                |
| **Target 模型**         | LFM2-1.2B (Q4_K_M)                | LFM2-1.2B (Q4_K_M)                            | 用于验证草稿质量。                                       |
| **内存占用 (Host)**       | **2580 MiB**                      | **2580 MiB**                                  | **两者完全相同**，SD 未增加额外显存/内存开销。                     |
| **内存占用 (CPU_REPACK)** | **482 MiB**                       | **482 MiB**                                   | 两者相同，内存结构一致。                                    |


# 关键发现

- 加速效果显著但受接受率限制：尽管接受率仅 28.5%，但由于草稿模型计算成本极低（350M vs 1.2B），生成速度提升 68%，加速比：1.68。

