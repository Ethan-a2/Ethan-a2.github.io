
[MNN/demo/exec at master · Ethan-a2/MNN](https://github.com/Ethan-a2/MNN/tree/master/demo/exec)

# build
```
cd path/to/MNN
mkdir build
cd build
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_DEMO=ON -DMNN_BUILD_CONVERTER=ON ..
make
```


# convert
```
./MNNConvert -f TF --modelFile model-mobilenet_v1_075.pb  --MNNModel model-mobilenet_v1_075.mnn --bizCode biz
```

# excute
```
./multiPose.out model-mobilenet_v1_075.mnn  multipose_input.png  pose.png
```


示例图片在这个路径:
```
../_static/images/start/multipose_input.png
```


参考:
https://mnn-docs.readthedocs.io/en/2.9.6/_sources/start/demo.md.txt

# 报错信息
```
CPU Group: [ 2  0  3  1 ], 800000 - 3500000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
        **Tensor shape**: 1, 225, 3, 225, 
Error for compute convolution shape, inputCount:3, outputCount:24, KH:3, KW:3, group:1
inputChannel: 225, batch:1, width:225, height:3. Input data channel may be mismatch with filter channel count
Compute Shape Error for MobilenetV1/add
Invalid Tensor, the session may not be ready
Can't run session because not resized
main, 381, cost time: 0.004000 ms
main, 405, cost time: 0.001000 ms
```


# 解决办法一
将输入的图像转换为tf模块的NHWC的格式。

## patch
```
diff --git a/demo/exec/multiPose.cpp b/demo/exec/multiPose.cpp
index 0b6a1a43..514c8961 100644
--- a/demo/exec/multiPose.cpp
+++ b/demo/exec/multiPose.cpp
@@ -334,7 +334,7 @@ int main(int argc, char* argv[]) {
     auto input = mnnNet->getSessionInput(session, nullptr);
 
     if (input->elementSize() <= 4) {
-        mnnNet->resizeTensor(input, {1, 3, targetHeight, targetWidth});
+        mnnNet->resizeTensor(input, {1, targetHeight, targetWidth, 3});
         mnnNet->resizeSession(session);
     }
```


## 原因
- 问题出在 MNN 模型文件期望的输入数据排布格式与您在 C++ 代码中提供的数据排布格式不一致。原始的 TensorFlow 模型是 NHWC，而您的 .mnn 模型转换后很可能也保留了对 NHWC 输入的期望。您的原代码提供了 NCHW 输入，导致模型内部对输入张量尺寸的误解，进而引发了后续层的计算错误。修改后的代码提供了 NHWC 输入，匹配了模型的期望，因此解决了问题。
- NCHW (Batch, Channel, Height, Width): 这是许多深度学习框架（如 PyTorch）和硬件后端（尤其是NVIDIA GPU）常用的格式。通道维是第二个维度。
- NHWC (Batch, Height, Width, Channel): 这是 TensorFlow 框架常用的格式。通道维是最后一个维度。
- 原始模型（TensorFlow .pb）通常是 NHWC 格式的。


# 解决办法二
在模块转换阶段，将TF要求的NCHW转换为MNN的NHWC
```
./MNNConvert -f TF --modelFile model-mobilenet_v1_075.pb  --MNNModel model-mobilenet_v1_075.mnn --bizCode biz --keepInputFormat false
```

即在最后添加--keepInputFormat false参数。


# reference
- https://github.com/alibaba/MNN/issues/2859
