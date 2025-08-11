# MobileSAM-ONNX-Runtime

部署MobileSAM（TinyViT）于  ONNX runtime上

## 编译

1. 环境准备：需要安装opencv以及onnx runtime，请参考网页教程，本onnx模型采用的onnx runtime版本为1.16.3，请安装相同对应版本的ort以避免bug。
2. 在 vs studio上，链接外部库与依赖，可参考[【深度学习】【onnxruntime】C++调用onnx_c++调用onnx模型-CSDN博客](https://blog.csdn.net/yangyu0515/article/details/142057357)
3. 将opencv和onnx runtime的动态链接库复制到编译生成的exe同目录。

## 运行

example

```c++
./ONNXRuntime.exe --backbone "path\to\mobile_sam_encoder.onnx" --model "path\to\mobile_sam.onnx" --img "path\to\cat.jpg" --points "324,432" --labels "1" --device "cpu"
```
<img width="1922" height="1264" alt="1" src="https://github.com/user-attachments/assets/ab27bc4a-58c9-4bce-a97e-8aad223ea17c" />

结果应当为：

```c++
(base) PS F:\Projects\C++\onnx\ONNXRuntime\x64\Release> ./ONNXRuntime.exe --backbone "F:\Projects\C++\onnx\ONNXRuntime\weights\mobile_sam_encoder.onnx" --model "F:\Projects\C++\onnx\ONNXRuntime\weights\mobile_sam.onnx" --img "F:\Projects\C++\onnx\ONNXRuntime\weights\cat.jpg" --points "324,432" --labels "1" --device "cpu"
Backbone path F:\Projects\C++\onnx\ONNXRuntime\weights\mobile_sam_encoder.onnx
model path F:\Projects\C++\onnx\ONNXRuntime\weights\mobile_sam.onnx
img path F:\Projects\C++\onnx\ONNXRuntime\weights\cat.jpg
Point prompts 324,432
labels 1
device cpu
Backbone target size: 1024x1024
Mask model inputs (6):
  image_embeddings
  point_coords
  point_labels
  mask_input
  has_mask_input
  orig_im_size
Mask model outputs:
  masks
  iou_predictions
  low_res_masks
Saved overlay to mobile_sam_cpp_out.png
推理耗时: 2264.47 ms
CPU 占用率: 1.37993 %
内存占用: 520.223 MB
```



## TODO
- ONNX 模型导出注意
- Linux ort 部署
- 支持 bbox prompt
- 集成到SAM-Linux WebServer



