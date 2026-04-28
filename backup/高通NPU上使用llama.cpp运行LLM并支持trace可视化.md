# 效果图

<img width="1699" height="382" alt="Image" src="https://github.com/user-attachments/assets/fad69b23-5306-4978-9a6f-676b6be46550" />


# environment
- xiaomi 15，Snapdragon® 8 Elite Mobile
- llama.cpp
- LLM模型:LFM2


# build
- 完整编译运行：
[llama.cpp/docs/backend/hexagon/README.md at master · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/hexagon/README.md)

将必要的文件推到手机里面：
```
adb shell mkdir -p /data/local/tmp/gguf
adb push LFM2-1.2B-Q4_0.gguf /data/local/tmp/gguf
adb push pkg-adb/llama.cpp /data/local/tmp/
adb push surfing.txt /data/local/tmp/llama.cpp
adb push LFM2-350M-Q4_0.gguf /data/local/tmp/gguf
```

surfing.txt:
```
1+1=?
```


# itrace
- 打上使能itrace的patch，或直接使用这个分支进行编译:
	- https://github.com/Ethan-a2/llama.cpp/tree/itrace
- 单独编译ggml-hexagon运行:
```
cmake --build build-snapdragon --target ggml-hexagon --verbose
adb push ./build-snapdragon/bin/libggml-hexagon.so /data/local/tmp/llama.cpp/lib/libggml-hexagon.so
TRACE=1 M=LFM2-1.2B-Q4_0.gguf D=HTP0 ./scripts/snapdragon/adb/run-cli.sh -no-cnv -p \"1+1=?\"
adb pull /data/local/tmp/itrace_results
```

- 浏览器打开: https://ui.perfetto.dev/
- 选择左边的: open trace file
- 选择: itrace_results/json/itrace_output.json



# PR
- 创建的PR:
- https://github.com/ggml-org/llama.cpp/pull/17837


# 完整trace及日志

[itrace_results.zip](https://github.com/user-attachments/files/24009818/itrace_results.zip)

[cli.log](https://github.com/user-attachments/files/24009873/cli.log)

# reference
- [llama.cpp/docs/backend/hexagon/README.md at master · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/hexagon/README.md)
