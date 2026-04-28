# 准备
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- 显存至少需要12G。在AutoDL上租一台RTX 2080 Ti x2 * 1卡，22G显存，大概1元/小时。整体费用5块。其它GPU平台类似。
- 小米15。小米12-8Gen1及之后的应该都行。
- QNN：2.36.0.250627。2.31的版本应该就行。
- AIMET：1.34.0。必须这个版本。。有点坑。使用QPM安装
- Ubuntu 22.04、Python 3.10
- 120GB总磁盘空间
- 量化后的模型已上传到huggingface，可以直接下载使用
	- https://huggingface.co/Eddie-L/Llama-3.2-1B-Instruct-Genie-QNN-NPU-8Gen4


## uv管理python环境
- 因为AIMET对版本有强要求。所以使用uv来管理python环境，方便复用。
```
cd /opt/qcom/aistack/tutorials/; 
uv venv py10 -p /usr/bin/python3.10
source py10/bin/activate

uv run python --version
cd /opt/qcom/aistack/aimet/1.34.0.44

uv pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
uv pip install  *.whl
```

- 关键库的版本。
```
accelerate                0.33.0
aimet                     1.34.0.0.207.0.44+torch.gpu
aimetcommon               1.34.0.0.207.0.44+torch.gpu
aimettorch                1.34.0.0.207.0.44+torch.gpu
numpy                     1.23.5
peft                      0.15.0
tokenizers                0.19.0
torch                     2.1.2+cu121
torchvision               0.16.2+cu121
transformers              4.43.2
```


# 1.模型适配与量化
- 第一步参考教程中的脚本，在GPU服务器上运行。关键点如下
- 模型适配
	- 线性层转卷积层 (`Linear` to `Conv1d`)
	- Attention层只输出当前token新生成的K和V，而不是整个更新后的KV序列。
	- 将动态计算（mask, pos_embed）变为静态输入。把attention_mask和positional_embedding提前准备好。
*   高级PTQ技术:
	*   混合精度 (Mixed Precision): 策略性地使用16-bit和8-bit量化。8bit KV-Cache, Concat优化
	*   SeqMSE: 最小化误差的权重参数优化方法。
	*   特定规则 (Custom Rules): 针对`MatMul`和`Concat`的性能优化。
	*   PPL（Perplexity）是评估模型质量的核心指标，贯穿始终。
- 输出为ONNX PTQ模型、AIMET encodings文件。再加test_vectors和tokenizer，用于离线推理。这一步生成的文件总大小大概50GB。刚好能在AutoDL的数据盘放下。。

```
├── onnx
│   ├── llama32_1b.encodings
│   ├── llama32_1b.onnx
│   ├── llama32_1b.pth
│   ├── llama32_1b_torch.encodings
│   ├── lm_head_conv_Conv.weight
│   ├── model_embed_tokens_Gather.weight
│   ├── model_layers_0_mlp_down_proj_conv_Conv.weight
│   ├── model_layers_0_mlp_gate_proj_conv_Conv.weight
│   ├── model_layers_0_mlp_up_proj_conv_Conv.weight
│   ├── model_layers_0_self_attn_k_proj_conv_Conv.weight
│   ├── model_layers_0_self_attn_o_proj_conv_Conv.weight
│   ├── model_layers_0_self_attn_q_proj_conv_Conv.weight
...
├── output
│   └── tokenizer
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── tokenizer.json
└── test_vectors
    ├── fp_0.pkl
    ├── layer_output_name_order.json
    └── qt_0.pkl
```

- llama32_1b.onnx: 通过脚本从 Hugging Face型转换而来。这个转换过程会将模型的 Pythonic 定义转换为一个静态的、语言无关的计算图。
- llama32_1b.encodings: 后训练量化 (PTQ) 时生成。使用一批校准数据（calibration data）来分析模型中每一层激活值的分布范围，并计算出最佳的量化参数，最后将这些参数保存在这个 `.encodings` 文件中。它记录了模型中每一层的量化参数，比如缩放因子 (scale) 和零点 (zero-point)。它就像一张“量化说明书”，告诉编译器如何将 32 位浮点数精确地映射到 8 位整数，同时最大程度地保留模型精度。
- fp_0.pkl: pickle文件，里面存储了用于模型校准和验证的输入数据。通常它是一个 NumPy 对象数组 (`numpy object array`)。这些数据是“有代表性”的，能够反映真实场景下的输入情况。


# 2.生成 QNN模型
- 这一步不依赖GPU，在本地笔记本或台式机上运行即可。
- 模型分割 (Split)
- attention block MHA 到 SHA 的转换 (Multi-Head to Single-Head Attention)
- 使用qairt-converter，将onnx转为QNN.
- 使用qairt-quantizer生成QNN量化库文件.
- 处理 Prompt（上下文）AR-128 和 逐个生成 Token AR-1这两部分权重共享。
- 使用 `qnn-context-binary-generator` 将 `ar1` 和 `ar128` 两个模型的量化 DLC 打包成weight_sharing_model_1_of_1.serialized.bin。


# 3.在手机上运行

- 所有必备的文件如下
```
├── to_device
│   ├── genie-t2t-run
│   ├── htp_backend_ext_config.json
│   ├── htp-model-config-llama32-1b-gqa.json
│   ├── libGenie.so
│   ├── libQnnHtpNetRunExtensions.so
│   ├── libQnnHtp.so
│   ├── libQnnHtpV79Skel.so
│   ├── libQnnHtpV79Stub.so
│   ├── libQnnSystem.so
│   ├── lprompt_1024.txt
│   ├── lprompt_4096.txt
│   ├── models
│   │   └── weight_sharing_model_1_of_1.serialized.bin
│   └── tokenizer.json
```

- 手机上执行的命令
```
adb shell
mkdir -p /data/local/tmp/llama3_2_assets
cd /data/local/tmp/llama3_2_assets
export LD_LIBRARY_PATH=$PWD
export ADSP_LIBRARY_PATH=$PWD
./genie-t2t-run -c ./htp-model-config-llama32-1b-gqa.json -p '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlan a 5 day trip to London for 4 people.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

./genie-t2t-run -c ./htp-model-config-llama32-1b-gqa.json --prompt_file lprompt_1024.txt
```


# 配置改动
- 模型名字改成这个meta-llama/Llama-3.2-1B-Instruct
- nsp_target = NspTargets.Android.GEN4，小米15是8Gen4。


# 性能

```
./genie-t2t-run -c ./htp-model-config-llama32-1b-gqa.json --prompt_file lprompt_1024.txt --profile perf.json
```

- 使用--profile参数生成性能数据，核心指标如下：
- token-generation-rate (Token 生成速率): 38.52 toks/sec。模型生成回复的速率（每秒生成的 Token 数）。这是衡量模型吞吐量的关键指标。
- prompt-processing-rate: 1547.98 toks/sec。模型处理输入提示的速率（每秒处理的 Token 数）。
- time-to-first-token (首 Token 延迟): 531665 (微秒) ≈ 0.53 秒。从发送查询到接收到第一个响应 Token 的时间。这是衡量模型响应速度的关键指标。
- duration: 5726945 (微秒) ≈ 5.73 秒。整个查询-响应周期的总耗时。
- num-prompt-tokens: 823。输入给模型的提示（Prompt）的 Token 数量。
- num-generated-tokens: 200。模型生成的回复的 Token 数量。
- token-generation-time: 5191579 (微秒) ≈ 5.19 秒。纯粹用于生成 200 个 Token 的时间。可以验证：总耗时 ≈ 首Token延迟 + 生成时间 (5.73s ≈ 0.53s + 5.19s)。

- perf.json文件内容
```
{
  "header": {
    "header_version": {
      "major": 0,
      "minor": 1,
      "patch": 0
    },
    "version": {
      "major": 0,
      "minor": 1,
      "patch": 0
    },
    "artifact_type": "GENIE_PROFILE"
  },
  "metadata": {
    "timestamp": 391862341312
  },
  "components": [
    {
      "name": "dialog0",
      "type": "dialog",
      "events": [
        {
          "type": "GenieDialog_create",
          "duration": 1124864,
          "start": 391862341614,
          "stop": 391863466478,
          "init-time": {
            "value": 1124713,
            "unit": "us"
          }
        },
        {
          "type": "GenieDialog_query",
          "duration": 5726945,
          "start": 391863466531,
          "stop": 391869193476,
          "num-prompt-tokens": {
            "value": 823,
            "unit": ""
          },
          "prompt-processing-rate": {
            "value": 1547.9876708984375,
            "unit": "toks/sec"
          },
          "time-to-first-token": {
            "value": 531665,
            "unit": "us"
          },
          "num-generated-tokens": {
            "value": 200,
            "unit": ""
          },
          "token-generation-rate": {
            "value": 38.52525329589844,
            "unit": "toks/sec"
          },
          "token-generation-time": {
            "value": 5191579,
            "unit": "us"
          }
        },
        {
          "type": "GenieDialog_free",
          "duration": 32347,
          "start": 391869193478,
          "stop": 391869225825
        }
      ]
    }
  ]
}
```


# autoDL踩过的坑
- 无卡开机内存很小，很容易OOM，并且还看不任何报错。用来传模型文件倒是可以。
- 非高峰期网速大概15MB/s。第一步生成的模型文件传了大概40分钟。。
- 50GB磁盘空间不足导致量化转换失败。
- 晚上基本卡的用了不。


# 备注
- 完整的模型文件生成目录及python库的版本见附件。

[pip.txt](https://github.com/user-attachments/files/21565393/pip.txt)
[example1_tree.log](https://github.com/user-attachments/files/21565391/example1_tree.log)
[example2_tree.log](https://github.com/user-attachments/files/21565392/example2_tree.log)

# references
- [Deepseekr1 部署在Honor100pro的NPU - 知乎](https://zhuanlan.zhihu.com/p/20510640906)
- [高通AIStack(2)-NPU部署Llama2-教程下载 - 知乎](https://zhuanlan.zhihu.com/p/704795464)

