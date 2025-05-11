



# 环境准备 
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1060        Off | 00000000:01:00.0 Off |                  N/A |
| N/A   62C    P8               3W /  78W |   2426MiB /  6144MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A       958      G   /usr/lib/xorg/Xorg                            4MiB |
|    0   N/A  N/A      3029      C   /usr/local/bin/ollama                      2418MiB |
+---------------------------------------------------------------------------------------+
```


```
ollama run qwen3:1.7b
pip install evalscope
```


# 运行
```
evalscope perf \
    --model qwen3:1.7b \
    --url "http://192.168.1.3:11434/v1/chat/completions" \
    --parallel 5 \
    --number 20 \
    --api openai \
    --dataset openqa \
    --stream
```


# 结果
```
"Time taken for tests (s)": 288.8133,
"Number of concurrency": 5,
"Total requests": 10,
"Succeed requests": 10,
"Failed requests": 0,
"Output token throughput (tok/s)": 36.605,
"Total token throughput (tok/s)": 37.5848,
"Request throughput (req/s)": 0.0346,
"Average latency (s)": 112.7879,
"Average time to first token (s)": 22.1013,
"Average time per output token (s)": 0.0865,
"Average input tokens per request": 28.3,
"Average output tokens per request": 1057.2,
"Average package latency (s)": 0.0865,
"Average package per request": 1048.3
```






# reference
- [Best Practices for Evaluating the Qwen3 Model | EvalScope](https://evalscope.readthedocs.io/en/latest/best_practice/qwen3.html)
- [量化模型效果评估 - Qwen](https://qwen.readthedocs.io/zh-cn/latest/getting_started/quantization_benchmark.html)
- [How Smart is Your AI? Full Assessment of IQ and EQ! | EvalScope](https://evalscope.readthedocs.io/en/latest/best_practice/iquiz.html)
- 