

```
git lfs install
git clone https://www.modelscope.cn/qwen/Qwen2-0.5B-Instruct.git

或者
huggingface-cli download taobao-mnn/Qwen2-0.5B-Instruct-MNN --local-dir .//Qwen2-0.5B-Instruct-MNN
```

# export

```
pip3 install torch torchvision torchaudio --force-reinstall
pip install numpy==1.26.4
python llmexport.py --path /media/code/mnn/Qwen2-0.5B-Instruct --export mnn



```



# build
```
cmake -DMNN_BUILD_DEMO=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_BENCHMARK=true -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true ..

make -j16
```



# run
```
./llm_demo /media/code/mnn/MNN/transformers/llm/export/model/config.json
```


# issue
聊天效果很差。。


使用llama.cpp，效果类似，都没有手机上的效果好。



```
huggingface-cli download Qwen/Qwen2-0.5B-Instruct-GGUF qwen2-0_5b-instruct-q5_k_m.gguf
huggingface-cli download Qwen/Qwen2-0.5B-Instruct-GGUF qwen2-0_5b-instruct-q5_k_m.gguf --local-dir .


/media/code/mnn/llama.cpp/build/bin/llama-cli -m ./qwen2-0_5b-instruct-q5_k_m.gguf -p "You are a helpful assistant" -cnv
```



- https://huggingface.co/taobao-mnn/Qwen2-0.5B-Instruct-MNN
- https://blog.csdn.net/j_kkko/article/details/139883432
- https://modelscope.cn/models/MNN/Qwen2-0.5B-Instruct-MNN

