# 设备
- K70 Pro Adreno 750 (Snapdragon 8 Gen 3)
- NDK26
- Adreno使用opencl编译


# 模型
- gemma-3n-E2B-it-UD-Q4_K_XL.gguf


# Android CPU上运行
```
ndk-r26d:
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_C_FLAGS="-march=armv8.7a" -DCMAKE_CXX_FLAGS="-march=armv8.7a" -DGGML_OPENMP=OFF  -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -B build-android
cmake --build build -j16
cmake --install build-android --prefix ../install

cd ../install
adb shell "mkdir /data/local/tmp/llama.cpp"
adb push install/. /data/local/tmp/llama.cpp/
adb push {model}.gguf /data/local/tmp/llama.cpp/
adb shell "ls /data/local/tmp/llama.cpp"
adb shell
cd /data/local/tmp/llama.cpp/lib
export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib:$LD_LIBRARY_PATH
./llama-server -m ../gemma-3n-E2B-it-UD-Q4_K_XL.gguf --top-p 0.95 --temp 0.7 --frequency-penalty 0 --presence-penalty 0 -n 40960 -s -1 --dynatemp-range 0 --dynatemp-exp 1 --top-k 40 --min-p 0.05 --typical 1 --repeat-last-n 64 --repeat-penalty 1 --mirostat 0 --mirostat-ent 5 --mirostat-lr 0.1  -c 40960 -np 1 -t -1 -ngl 14 --host 0.0.0.0 --verbose
```


# Adreno上运行
```
Adreno:

Install OpenCL Headers and Library
git clone https://github.com/KhronosGroup/OpenCL-Headers && \
cd OpenCL-Headers && \
cp -rf ./CL/ $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include

git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader && \
cd OpenCL-ICD-Loader && \
mkdir build_ndk26 && cd build_ndk26
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DOPENCL_ICD_LOADER_HEADERS_DIR=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=24 -DANDROID_STL=c++_shared
cp libOpenCL.so $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android

Build llama.cpp:
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a  -DANDROID_PLATFORM=android-28  -DBUILD_SHARED_LIBS=OFF -DGGML_OPENCL=ON  -DLLAMA_CURL=OFF
ninja -j16
DESTDIR=../install-opencl/ ninja install


cd ../install-opencl/usr/loca/
adb shell "mkdir /data/local/tmp/llama.cpp-opencl"
adb push . /data/local/tmp/llama.cpp-opencl/
adb push {model}.gguf /data/local/tmp/llama.cpp-opencl/
adb shell "ls /data/local/tmp/llama.cpp-opencl"
adb shell
cd /data/local/tmp/llama.cpp-opencl/lib
export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp-opencl/lib:$LD_LIBRARY_PATH
./llama-server -m ../gemma-3n-E2B-it-UD-Q4_K_XL.gguf --top-p 0.95 --temp 0.7 --frequency-penalty 0 --presence-penalty 0 -n 40960 -s -1 --dynatemp-range 0 --dynatemp-exp 1 --top-k 40 --min-p 0.05 --typical 1 --repeat-last-n 64 --repeat-penalty 1 --mirostat 0 --mirostat-ent 5 --mirostat-lr 0.1  -c 40960 -np 1 -t -1 -ngl 14 --host 0.0.0.0 --verbose
```


# Reference
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md
- [google-ai-edge](https://github.com/google-ai-edge)

# issue
- 用CPU基本与Google官方的gallery性能一致。GPU性能不及gallery
