import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import os

# 加载预训练的ResNet50模型
model = models.resnet50(weights=None) #把权重文件下载到当前文件夹会更快
model.load_state_dict(torch.load('../resnet50-19c8e357.pth'))  # 加载训练好的模型权重
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 特征提取函数
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    # return features.numpy().flatten()
    return features

class ImageProcessor:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.output_path = os.path.join(input_folder, "processed_data.npz")
        self.features = []
        # 预留处理接口（用户可自定义实现）
        self.process_image = self._default_processor
        
    def _default_processor(self, image_path):
        """示例处理函数（需替换为实际逻辑）"""
        img = Image.open(image_path)
        return extract_features(image_path)  
    

    def process_folder(self):
    # 用于存储子文件夹路径及其对应的特征向量
        subfolder_features = {}
        
        # 遍历所有子文件夹
        for root, _, files in os.walk(self.input_folder):
            # 计算当前子文件夹相对于输入根目录的路径
            rel_subfolder = os.path.relpath(root, self.input_folder)
            if rel_subfolder == ".":
                rel_subfolder = ""  # 根目录标记为空字符串
            
            # 筛选并排序当前子文件夹中的图片
            img_files = [
                f for f in files 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not img_files:
                continue  # 跳过无图片的子文件夹
            
            # 按文件名数字排序（例如 "001.jpg" 取数字部分排序）
            # img_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0])))
            
            # 处理当前子文件夹内的所有图片
            
            for f in tqdm(img_files, desc=f"Processing {rel_subfolder or 'root'}"):
                try:
                    img_path = os.path.join(root, f)
                    feature = self.process_image(img_path)
                    self.features.append(feature)
                except Exception as e:
                    print(f"Error in {os.path.join(rel_subfolder, f)}: {str(e)}")
            

        
        # 保存结果
        if 1:
            # 将字典拆分为路径列表和特征列表
            # subfolders = list(subfolder_features.keys())
            feature_arrays = np.vstack(self.features)
            
            # np.savez_compressed(
            #     self.output_path,
            #     subfolders=np.array(subfolders, dtype=str),
            #     data=np.array(feature_arrays, dtype=object)  # 允许不同子文件夹的特征数量不同
            # )

            np.savez_compressed(
                self.output_path,
                data=feature_arrays  
            )

        else:
            print("No valid features found in any subfolders.")
        
        return self.output_path


    

# # 初始化处理器
processor = ImageProcessor("./")
    
    
# # 执行处理
output_path = processor.process_folder()
    
# # 读取数据
loaded = np.load("processed_data.npz", allow_pickle=True)


# 通过键名获取数据

feature_arrays = loaded["data"]    # 特征向量对象数组
print(feature_arrays.shape)

