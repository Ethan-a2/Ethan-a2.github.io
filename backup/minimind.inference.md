
# download
```
git clone https://github.com/jingyaogong/minimind.git
huggingface-cli download --local-dir MiniMind2
```

# test
```
python eval_model.py --load 1 --model_mode 2

cd scripts
streamlit run web_demo.py

```

# 使用llama.cpp推理
在llama.cpp中打上如下patch:
```
diff --git a/convert_hf_to_gguf.py b/convert_hf_to_gguf.py
index bf6bc683..5ea853b6 100755
--- a/convert_hf_to_gguf.py
+++ b/convert_hf_to_gguf.py
@@ -808,7 +808,8 @@ class TextModel(ModelBase):
             logger.warning(f"** chkhsh:  {chkhsh}")
             logger.warning("**************************************************************************************")
             logger.warning("\n")
-            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")
+            res = "smollm"
+            #raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")
 
         logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
         logger.debug(f"chkhsh: {chkhsh}")
```

```
python convert_hf_to_gguf.py ../minimind/MiniMind2/

./build/bin/llama-quantize ../minimind/MiniMind2/MiniMind2-109M-F16.gguf ../minimind/MiniMind2/Q4-MiniMind2.gguf Q4_K_M

./build/bin/llama-cli -m ../minimind/MiniMind2/MiniMind2-109M-F16.gguf --chat-template chatml

```

# 使用ollama部署
新建minimind.modelfile：
```
FROM ./MiniMind2-109M-F16.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
```

加载模型并命名为minimind2
```
ollama create -f minimind.modelfile minimind2
```

启动推理

```
ollama run minimind2
```

# reference
[jingyaogong/minimind: 🚀🚀 「大模型」2小时完全从0训练26M的小参数GPT！🌏 Train a 26M-parameter GPT from scratch in just 2h!](https://github.com/jingyaogong/minimind#vllm%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E6%9C%8D%E5%8A%A1)
