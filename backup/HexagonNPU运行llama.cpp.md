本文将详细介y绍如何在 Android 设备上，利用 `llama.cpp` 框架，结合高通神经网络处理单元 (QNN) 的硬件加速能力，部署并运行 Qwen 1.5 模型。

本教程将涵盖模型转换、交叉编译 `llama.cpp`、文件传输到设备以及最终运行推理的完整流程。

# 前置条件

在开始之前，请确保你已经准备好以下环境和工具：

1.  **操作系统**: Linux (推荐 Ubuntu) 或 macOS。Windows 用户可以使用 WSL2。
2.  **Git**: 用于克隆 `llama.cpp` 仓库。
    ```bash
    sudo apt update && sudo apt install git # Linux
    ```
3.  **Python 3**: 及其包管理工具 `pip`。
    ```bash
    sudo apt install python3 python3-pip # Linux
    pip install transformers sentencepiece accelerate # 用于模型下载和转换
    ```
4.  **CMake**: 版本 3.15 或更高。
    ```bash
    sudo apt install cmake # Linux
    ```
5.  **Android NDK**: 用于编译 Android 代码。建议下载最新稳定版。
    *   下载地址: [Android NDK download](https://developer.android.com/ndk/downloads)
    *   解压后，请设置环境变量 `ANDROID_NDK_ROOT` 指向 NDK 的根目录。例如：
        ```bash
        export ANDROID_NDK_ROOT=/path/to/android-ndk-rXX (替换为你的 NDK 路径，例如 ~/Android/android-ndk-r26c)
        # 建议将其添加到 ~/.bashrc 或 ~/.zshrc 以便永久生效
        ```
6.  **Qualcomm QNN SDK**: 如果你想启用 QNN 加速，这是必需的。你需要从高通开发者网站或通过合作渠道获取。
    *   下载 QNN SDK 并解压。
    *   设置环境变量 `QNN_SDK_ROOT` 指向 QNN SDK 的根目录。例如：
        ```bash
        export QNN_SDK_ROOT=/path/to/qnn_sdk (替换为你的 QNN SDK 路径)
        # 建议将其添加到 ~/.bashrc 或 ~/.zshrc 以便永久生效
        ```
    *   **重要**: QNN 加速仅在支持 QNN 的高通芯片设备上有效。例如，骁龙 800 系列的部分 SoC。
7.  **ADB (Android Debug Bridge)**: 用于与 Android 设备通信。通常随 Android SDK Platform-Tools 一起安装。
    ```bash
    sudo apt install android-tools-adb # Linux
    ```
8.  **一台 Android 设备**: 已开启开发者模式和 USB 调试。
    *   验证设备连接：`adb devices`

# 步骤一：模型导出为 GGUF 格式

`llama.cpp` 使用其特有的 GGUF 格式来存储模型，这种格式支持内存映射，能高效加载模型。我们将下载原始的 Hugging Face Qwen 1.5 模型，然后将其转换为 GGUF 格式。

1.  **克隆 `llama.cpp` 仓库**:
    ```bash
    git clone https://github.com/chraac/llama.cpp.git
    cd llama.cpp
    ```

2.  **下载 Qwen 1.5 模型**:
    我们将模型下载到 `llama.cpp` 仓库根目录下的 `models/Qwen1.5-0.5B` 文件夹中。
    ```bash
    mkdir -p models/Qwen1.5-0.5B
    huggingface-cli download Qwen/Qwen1.5-0.5B --local-dir models/Qwen1.5-0.5B
    ```

3.  **转换为 GGUF 格式**:
    `llama.cpp` 仓库中提供了一个 `convert.py` 脚本用于模型转换。
    *   **注意**: 原始文件中使用的是 `convert_hf_to_gguf.py`，这可能是一个自定义脚本或旧版本的名称。`llama.cpp` 官方的转换脚本通常是 `convert.py`。此处我们统一使用 `convert.py`。
    ```bash
    python3 convert.py models/Qwen1.5-0.5B/ --outfile qwen1.5-0.5B_fp32.gguf --outtype f32
    ```
    *   `--outfile qwen1.5-0.5B_fp32.gguf`: 指定输出的 GGUF 文件名。
    *   `--outtype f32`: 指定输出模型的精度为 FP32。你也可以尝试其他量化类型，如 `q4_0`、`q8_0` 等，以减小模型大小和提高推理速度（需要后续额外的量化步骤）。

4.  **验证生成文件**:
    转换完成后，确认在 `llama.cpp` 根目录或你指定的输出路径下生成了 `qwen1.5-0.5B_fp32.gguf` 文件。
    ```bash
    ls -lh qwen1.5-0.5B_fp32.gguf
    ```
    它的大小应该与原始模型文件大小相似或略有不同。

# 步骤二：为 Android 交叉编译 llama.cpp (启用 QNN 加速)

我们将编译 `llama.cpp` 为 Android 可执行文件，并启用 QNN 支持。

1.  **修改 CMakeLists.txt (可选，根据llama.cpp版本)**：
    在某些 `llama.cpp` 版本中，`LLAMA_CURL` 编译选项可能默认开启。如果你的 `llama.cpp` 版本中存在此选项，请将其改为 `OFF`，因为在 Android 环境下使用 `curl` 可能会引入额外的复杂性，且我们不需要从 URL 下载模型。
    在 `llama.cpp/CMakeLists.txt` 中找到并修改：
    ```cmake
    option(LLAMA_CURL       "llama: use libcurl to download model from an URL" OFF)
    ```
    *   **注意**: 较新版本的 `llama.cpp` 可能没有这个选项，或者默认就是 `OFF`。请根据你的实际代码库进行检查。

2.  **创建构建目录并配置 CMake**:
    在 `llama.cpp` 根目录下执行：
    ```bash
    mkdir build-android && cd build-android
    cmake -H.. -B. \
        -DBUILD_SHARED_LIBS=off \
        -DGGML_QNN_ENABLE_CPU_BACKEND=on \
        -DGGML_OPENMP=off \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_PLATFORM=android-31 \
        -DANDROID_NDK=$ANDROID_NDK_ROOT \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
        -DGGML_QNN=on \
        -DGGML_QNN_SDK_PATH="$QNN_SDK_ROOT" \
        -DCMAKE_BUILD_TYPE=Release
    ```
    *   `-H.. -B.`: 指定源码目录为父目录，构建目录为当前目录。
    *   `-DBUILD_SHARED_LIBS=off`: 编译为静态库，方便部署，避免依赖动态库查找问题。
    *   `-DGGML_QNN_ENABLE_CPU_BACKEND=on`: 启用 QNN 时，如果 QNN 后端不可用或出错，允许回退到 CPU 运行。
    *   `-DGGML_OPENMP=off`: 在 Android NDK 编译某些 OpenMP 特性时可能出现兼容性问题，此处关闭。
    *   `-DANDROID_ABI="arm64-v8a"`: 指定目标 ABI 为 ARMv8-A 架构，这是目前主流 Android 设备的架构。
    *   `-DANDROID_PLATFORM=android-31`: 指定 Android API 级别，这里是 Android 12 (API 31)。请根据你的设备和 NDK 版本适当调整。
    *   `-DANDROID_NDK=$ANDROID_NDK_ROOT`: 指向 Android NDK 的根目录。
    *   `-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake`: 指定 Android NDK 的交叉编译工具链文件。
    *   `-DGGML_QNN=on`: 启用 QNN 后端支持。
    *   `-DGGML_QNN_SDK_PATH="$QNN_SDK_ROOT"`: 指向 QNN SDK 的根目录。
    *   `-DCMAKE_BUILD_TYPE=Release`: 构建 Release 版本，优化性能。

3.  **编译 `llama.cpp`**:
    配置成功后，执行编译命令：
    ```bash
    cmake --build . --config Release -- -j$(nproc)
    ```
    *   `-j$(nproc)`: 使用所有可用的 CPU 核心进行并行编译，加速过程。

4.  **安装编译好的文件**:
    为了方便后续 `adb push`，我们将编译好的可执行文件和库安装到一个临时目录。
    ```bash
    # (确保当前位于 build-android 目录下)
    DESTDIR=../install-qnn/ make install
    ```
    *   `DESTDIR=../install-qnn/`: 指定安装目标目录为 `llama.cpp` 根目录下的 `install-qnn` 文件夹。
    *   `make install`: 执行安装目标。这会将 `llama.cpp` 的可执行文件（如 `llama-cli` 或 `main`）复制到 `../install-qnn/usr/local/bin/`，将相关库复制到 `../install-qnn/usr/local/lib/`。

5.  **验证安装文件**:
    回到 `llama.cpp` 根目录，检查 `install-qnn` 文件夹的内容。
    ```bash
    cd ../ && ls -RF install-qnn/
    # 你应该能看到类似 install-qnn/usr/local/bin/llama-cli (或 main) 等文件
    ```

# 步骤三：将模型和可执行文件推送到 Android 设备

现在，我们将编译好的可执行文件、QNN 运行时库（如果静态链接不完全）和 GGUF 模型文件传输到 Android 设备。

1.  **确认设备连接**:
    ```bash
    adb devices
    ```
    确保你的设备处于 `device` 状态。

2.  **在设备上创建目标目录**:
    我们将在 `/data/local/tmp/llama.cpp-qnn/` 下创建目录结构。`/data/local/tmp` 是一个用户可写入的临时目录。
    ```bash
    adb shell "mkdir -p /data/local/tmp/llama.cpp-qnn/bin"
    adb shell "mkdir -p /data/local/tmp/llama.cpp-qnn/lib"
    ```

3.  **推送编译好的执行文件和库**:
    进入 `install-qnn` 目录下，将 `bin` 和 `lib` 目录推送到设备。
    ```bash
    cd install-qnn/usr/local/
    adb push bin /data/local/tmp/llama.cpp-qnn/
    adb push lib /data/local/tmp/llama.cpp-qnn/
    ```
    *   **注意**: `lib` 目录可能包含 QNN 运行所需的动态库 (`libQNN_*.so`)。确保这些库也被推送到设备。

4.  **推送 GGUF 模型文件**:
    回到 `llama.cpp` 根目录，推送转换好的 GGUF 模型。
    ```bash
    cd ../../ # 回到 llama.cpp 根目录
    adb push qwen1.5-0.5B_fp32.gguf /data/local/tmp/llama.cpp-qnn/
    ```

5.  **验证文件是否已成功推送**:
    ```bash
    adb shell "ls -RF /data/local/tmp/llama.cpp-qnn/"
    ```
    你应该能看到 `bin/`、`lib/` 目录以及 `qwen1.5-0.5B_fp32.gguf` 文件。

# 步骤四：在 Android 设备上运行推理

现在所有文件都已就绪，我们可以在 Android 设备上运行 Qwen 1.5 模型进行推理了。

1.  **进入 Android 设备 Shell**:
    ```bash
    adb shell
    ```

2.  **导航到可执行文件目录**:
    ```bash
    cd /data/local/tmp/llama.cpp-qnn/bin/
    ```

3.  **设置动态库路径**:
    如果 `llama.cpp` 或 QNN 运行时依赖任何动态库，需要设置 `LD_LIBRARY_PATH` 环境变量，以便系统能找到这些库。
    ```bash
    export LD_LIBRARY_PATH=$PWD:/data/local/tmp/llama.cpp-qnn/lib:$LD_LIBRARY_PATH
    ```
    *   `$PWD`: 将当前目录（`bin`）添加到路径中。
    *   `/data/local/tmp/llama.cpp-qnn/lib`: 将 QNN 运行时库的路径添加到路径中。

4.  **赋予执行权限**:
    `llama-cli` (或 `main`) 可执行文件需要有执行权限。
    ```bash
    chmod +x ./llama-cli
    ```

5.  **运行模型推理**:
    现在，你可以执行 `llama-cli` 并指定模型文件。
    ```bash
    ./llama-cli -m ../qwen1.5-0.5B_fp32.gguf \
                -p "你好，请自我介绍一下。" \
                -n 256 \
                -ngl 99 \
                -t 4
    ```
    *   `-m ../qwen1.5-0.5B_fp32.gguf`: 指定模型文件路径。模型在 `llama-cli` 的父目录。
    *   `-p "你好，请自我介绍一下。"`: 指定输入提示。
    *   `-n 256`: 指定生成文本的最大长度（tokens）。
    *   `-ngl 99`: 这是一个关键参数，表示将模型的多少层放入 GPU (QNN) 运行。`99` 通常表示所有层都尝试加载到 GPU。如果你的设备支持 QNN，这会极大地加速推理。如果 QNN 初始化失败或不支持，`llama.cpp` 会回退到 CPU 运行。
    *   `-t 4`: 指定推理使用的 CPU 线程数。

    模型将开始加载并进行推理，你将在命令行看到模型的输出。第一次运行可能会因为 QNN SDK 的初始化而稍慢。
