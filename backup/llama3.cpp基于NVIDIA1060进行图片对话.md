
# 环境
显卡：NVIDIA GeForce GTX 1060
图片对话模型：SmolVLM-500M-Instruct



# 编译

```
sudo apt-get install libssl-dev libcurl4-openssl-dev

git clone https://github.com/ggml-org/llama.cpp.git
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
make -j16
```


在这里[Ollama GPU支持 - Nvidia和AMD GPU兼容性 | LlamaFactory | LlamaFactory](https://www.llamafactory.cn/ollama-docs/gpu.html#google_vignette)找到1060的架构是6.1，编译时指定-DCMAKE_CUDA_ARCHITECTURES=61


# 模型GGUF下载
```
export HF_HOME=/media/dataset/hf
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_TOKEN=xxx

huggingface-cli download ggml-org/SmolVLM-500M-Instruct-GGUF
huggingface-cli download ggml-org/SmolVLM-500M-Instruct-GGUF --local-dir .
```


# 运行llama3.cpp
```
export PATH"$PATH:/path/to/llama.cpp/build/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/llama.cpp/build/bin

llama-server -m ./SmolVLM-500M-Instruct-Q8_0.gguf --mmproj mmproj-SmolVLM-500M-Instruct-Q8_0.gguf --host 0.0.0.0 --port 8080 -ngl 100
```


输出日志:
```
User: Hello<end_of_utterance>
Assistant: Hi there<end_of_utterance>
User: How are you?<end_of_utterance>
Assistant:'
main: server is listening on http://0.0.0.0:8080 - starting the main loop
srv  update_slots: all slots are idle
```


- 用浏览器打开网址，比如 192.168.3.100:1080，即可以进行图片对话。1060的6G显卡，运行500M的模型，吐字飞快。
- 500M的模型，用CPU跑也不慢。另外，还可以尝试更小的200M的模型。


![Image](https://github.com/user-attachments/assets/457df473-481f-4a85-9adf-0c40f7754857)