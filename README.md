# Dog-Face-Recognition

<img src="https://raw.githubusercontent.com/GarfieldQAQ/Dog-Face-Recognition/refs/heads/main/ui/doggy.jpg" alt="drawing" width="200"/>
# 简述：
该工程实现了识别狗狗图像区别其是否在登记名册中(诗人握持
训练了抓取狗脸的yolo权重文件
用Resnet获取图片特征向量(1000维
使用PCA做数据降维
OneclassSVM 实现是否在库中的判断
SVM分类确定狗狗是哪一只狗
# 涉及的模型：
**YOLOv12 + ResNet + OneclassSVM + SVM**
# 主要工作流
## YOLOv12
### 前期准备
主要是在预训练的权重上进行训练，用yolodatebase文件夹下的10k余张狗狗照片训练YOLO抓取狗头，具体的训练结果以及权重保存在 ./runs/train/exp5 文件夹下，训练文件为 ./ 目录下的train.py 以及 data.yaml 文件，yolo.py 是使用训练好的模型处理扣取狗脸图片的脚本
### 处理原始图片获得狗脸数据集
调用yolo.py，详细使用方法见py文件
### Resnet
将狗脸数据集调整大小后直接输入预训练的Resnet模型获得1000维特征向量
### Oneclass SVM


# 文件tree:
在windows下tree命令并不能设置tree深度，具体文件树详见==tree.txt==
# 部署环境:
推荐使用虚拟环境
## conda创建py3.11虚拟环境
```bash
conda create --name "环境的名字" python=3.11
```
注意自己修改"环境名字"字段
## 激活虚拟环境
```bash
conda activate "环境的名字"
```
如果创建的时候将环境名字设置为 “test”
```bash
conda activate "test"
```
## 下载依赖
```bash
pip install -r requirements.txt
```
## 运行
进入ui文件夹，运行 window.py 即可
