
# environment
- Operating System: Ubuntu 22.04
- QNN SDK Version: 2.36.0.250627
- devices：xiaomi 15Ultra 16G内存, Snapdragon® 8 Elite Mobile
- genie-t2t-run

- 注意，[Qwen2-7B-Instruct - Qualcomm AI Hub](https://aihub.qualcomm.com/models/qwen2_7b_instruct)目前只支持Snapdragon® 8 Elite Mobile。

# Qwen2-7B-Instruct模型下载
- [Qwen2-7B-Instruct - Qualcomm AI Hub](https://aihub.qualcomm.com/models/qwen2_7b_instruct)


# QNN下载
- [Qualcomm Neural Processing SDK | Qualcomm Developer](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai)


# 量化配置文件
- qwen2_7b_instruct_quantized.json来自[aihub models run failure · Issue #15 · quic/ai-hub-apps](https://github.com/quic/ai-hub-apps/issues/15)
```
{
    "dialog": {
        "version": 1,
        "type": "basic",
        "context": {
            "version": 1,
            "size": 4096,
            "n-vocab": 152064,
            "bos-token": -1,
            "eos-token": 151645
        },
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.8,
            "top-k": 40,
            "top-p": 0.95
        },
        "tokenizer": {
            "version": 1,
            "path": "tokenizer.json"
        },
        "engine": {
            "version": 1,
            "n-threads": 3,
            "backend": {
                "version": 1,
                "type": "QnnHtp",
                "QnnHtp": {
                    "version": 1,
                    "use-mmap": true,
                    "spill-fill-bufsize": 0,
                    "mmap-budget": 0,
                    "poll": false,
                    "pos-id-dim": 64,
                    "cpu-mask": "0xe0",
                    "kv-dim": 128,
                    "rope-theta": 1000000,
                    "allow-async-init": false
                },
                "extensions": "htp_backend_ext_config.json"
            },
            "model": {
                "version": 1,
                "type": "binary",
                "binary": {
                    "version": 1,
                    "ctx-bins": [
                        "weight_sharing_model_1_of_4.serialized.bin",
                        "weight_sharing_model_2_of_4.serialized.bin",
                        "weight_sharing_model_3_of_4.serialized.bin",
                        "weight_sharing_model_4_of_4.serialized.bin"
                    ]
                }
            }
        }
    }
}

```



# tokenizer.json
- 来自[ai-hub-apps/tutorials/llm_on_genie at main · quic/ai-hub-apps](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie)
- 从Qwen2-7B-Instruct处下载对应的tokenizer.json




# 目录结构
- 所有需要的文件的完整目录结构如下:
```
qwen2/
├── genie_bundle
│   ├── genie-t2t-run
│   ├── libCalculator_skel.so
│   ├── libcalculator.so
│   ├── libGenie.so
│   ├── libhta_hexagon_runtime_qnn.so
│   ├── libhta_hexagon_runtime_snpe.so
│   ├── libPlatformValidatorShared.so
│   ├── libQnnChrometraceProfilingReader.so
│   ├── libQnnCpuNetRunExtensions.so
│   ├── libQnnCpu.so
│   ├── libQnnDspNetRunExtensions.so
│   ├── libQnnDsp.so
│   ├── libQnnDspV66CalculatorStub.so
│   ├── libQnnDspV66Stub.so
│   ├── libQnnGenAiTransformerCpuOpPkg.so
│   ├── libQnnGenAiTransformerModel.so
│   ├── libQnnGenAiTransformer.so
│   ├── libQnnGpuNetRunExtensions.so
│   ├── libQnnGpuProfilingReader.so
│   ├── libQnnGpu.so
│   ├── libQnnHtaNetRunExtensions.so
│   ├── libQnnHta.so
│   ├── libQnnHtpNetRunExtensions.so
│   ├── libQnnHtpOptraceProfilingReader.so
│   ├── libQnnHtpPrepare.so
│   ├── libQnnHtpProfilingReader.so
│   ├── libQnnHtp.so
│   ├── libQnnHtpV68CalculatorStub.so
│   ├── libQnnHtpV68Stub.so
│   ├── libQnnHtpV69CalculatorStub.so
│   ├── libQnnHtpV69Stub.so
│   ├── libQnnHtpV73CalculatorStub.so
│   ├── libQnnHtpV73Stub.so
│   ├── libQnnHtpV75CalculatorStub.so
│   ├── libQnnHtpV75Skel.so
│   ├── libQnnHtpV75.so
│   ├── libQnnHtpV75Stub.so
│   ├── libQnnHtpV79CalculatorStub.so
│   ├── libQnnHtpV79Stub.so
│   ├── libQnnIr.so
│   ├── libQnnJsonProfilingReader.so
│   ├── libQnnLpaiNetRunExtensions.so
│   ├── libQnnLpai.so
│   ├── libQnnLpaiStub.so
│   ├── libQnnModelDlc.so
│   ├── libQnnNetRunDirectV79Stub.so
│   ├── libQnnSaver.so
│   ├── libQnnSystem.so
│   ├── libQnnTFLiteDelegate.so
│   ├── libSnpeDspV66Stub.so
│   ├── libSnpeHta.so
│   ├── libSnpeHtpPrepare.so
│   ├── libSnpeHtpV68CalculatorStub.so
│   ├── libSnpeHtpV68Stub.so
│   ├── libSnpeHtpV69CalculatorStub.so
│   ├── libSnpeHtpV69Stub.so
│   ├── libSnpeHtpV73CalculatorStub.so
│   ├── libSnpeHtpV73Stub.so
│   ├── libSnpeHtpV75CalculatorStub.so
│   ├── libSnpeHtpV75Skel.so
│   ├── libSnpeHtpV75Stub.so
│   ├── libSnpeHtpV79CalculatorStub.so
│   ├── libSnpeHtpV79Stub.so
│   └── libSNPE.so
├── htp_backend_ext_config.json
├── qwen2_7b_instruct_quantized.json
├── tokenizer.json
├── weight_sharing_model_1_of_4.serialized.bin
├── weight_sharing_model_2_of_4.serialized.bin
├── weight_sharing_model_3_of_4.serialized.bin
└── weight_sharing_model_4_of_4.serialized.bin
```



# 万事俱备
- 将上面的模块文件，量化配置文件，qnn库全部push到手机。命令如下
```
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.36.0.250627
source $QNN_SDK_ROOT/bin/envsetup.sh

adb shell mkdir -p /data/local/tmp/qwen2/
adb push qwen2_7b_instruct_quantized.json /data/local/tmp/qwen2/
adb push tokenizer.json /data/local/tmp/qwen2/
adb push htp_backend_ext_config.json /data/local/tmp/qwen2/
adb push *.bin /data/local/tmp/qwen2/

cp $QNN_SDK_ROOT/lib/hexagon-v79/unsigned/* genie_bundle
cp $QNN_SDK_ROOT/lib/aarch64-android/* genie_bundle
cp $QNN_SDK_ROOT/bin/aarch64-android/genie-t2t-run genie_bundle
adb push genie_bundle/* /data/local/tmp/qwen2/
```


# 手机上运行genie-t2t-run
```
adb shell
cd /data/local/tmp/qwen2/
export LD_LIBRARY_PATH=$PWD
export ADSP_LIBRARY_PATH=$PWD

./genie-t2t-run -c qwen2_7b_instruct_quantized.json -p "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```
