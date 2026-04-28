
# 实例分割
转换时需要添加--keepInputFormat false参数，转为HCHW的格式。否则虽然没有报错，但是结果是黑图
[示例工程 — MNN-Doc 2.1.1 documentation](https://mnn-docs.readthedocs.io/en/latest/start/demo.html#id3)

```
./MNNConvert -f TFLITE --modelFile deeplabv3_257_mv_gpu.tflite --MNNModel deeplabv3_257_mv_gpu.mnn --bizCode biz  --keepInputFormat false
./segment.out deeplabv3_257_mv_gpu.mnn segment_input.png result.png

```


./GetMNNInfo deeplabv3_257_mv_gpu.mnn
```
CPU Group: [ 2  0  3  1 ], 800000 - 3500000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
Model default dimensionFormat is NHWC
Model Inputs:
[ sub_7 ]: dimensionFormat: NC4HW4, size: [ 1,3,257,257 ], type is float
Model Outputs:
[ ResizeBilinear_3 ]
Model Version: 3.1.4 
Model bizCode: biz
```


# 图像识别

```
git clone https://github.com/shicai/MobileNet-Caffe
./MNNConvert -f CAFFE --modelFile MobileNet-Caffe/mobilenet.caffemodel --prototxt MobileNet-Caffe/mobilenet_deploy.prototxt --MNNModel mobilenet.mnn --bizCode biz
./GetMNNInfo mobilenet.mnn

./pictureRecognition.out mobilenet.mnn ../demo/model/MobileNet/testcat.jpg
```


- 第一行表示识别出可能性最大的类别编号，在相应的 synset_words.txt 去查找对应的类别，如：demo/model/MobileNet/synset_words.txt
- 注意是结果是从0开始。
- 对应imageNet的1000个类别。

```
./pictureRecognition.out mobilenet.mnn ../demo/model/MobileNet/testcat.jpg
Can't open file:.cachefile
Load Cache file error.
CPU Group: [ 2  0  3  1 ], 800000 - 3500000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
Session Info: memory use 21.544098 MB, flops is 568.742310 M, backendType is 13, batch size = 1
input: w:224 , h:224, bpp: 3
origin size: 480, 360
For Image: ../demo/model/MobileNet/testcat.jpg
282, 0.248962
277, 0.156494
263, 0.148438
278, 0.121017
281, 0.026409
259, 0.024555
151, 0.023830
285, 0.021233
287, 0.017479
280, 0.014622
```

https://blog.csdn.net/weixin_43214408/article/details/120712878


# MNNV2Basic.out
```
./MNNV2Basic.out mobilenet.mnn
./MNNV2Basic.out mobilenet.mnn 10 1  //跑十次，打印输入、输出



./timeProfile.out mobilenet.mnn

./getPerformance.out

./mobilenetTest.out mobilenet.mnn cat.jpg

./winogradExample.out 3 3
```


- 打印每一层的输入和输出，然观察输入输出的文件大小变化，可以看到第56层的输入从1615k跳变成了1k,初步断定是第56层出了问题，


# benchmark

```
cmake -DMNN_BUILD_DEMO=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_BENCHMARK=true ..

./benchmark.out ../benchmark/models/ 10
```
