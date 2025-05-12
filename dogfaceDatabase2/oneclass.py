from sklearn.svm import OneClassSVM
import numpy as np
import cloudpickle
import os
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.models as models
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
loaded = np.load(".npz", allow_pickle=True)
feature_arrays = loaded["data"]    # 特征向量对象数组
print(feature_arrays.shape)


#训练One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)  # nu控制异常值比例
oc_svm.fit(feature_arrays)

# 计算所有训练样本的决策函数值
decision_values = oc_svm.decision_function(feature_arrays)
print(decision_values)
threshold = np.percentile(decision_values, 1)  # 假设允许5%的误拒率
print(threshold)


with open('oneclass_svm_model2.pkl', 'wb') as f:
    cloudpickle.dump(oc_svm, f)
# # 2. 保存模型到 .pkl 文件


# 3. 加载模型
with open('oneclass_svm_model2.pkl', 'rb') as f:
    oneclass_svm_model = cloudpickle.load(f)
# 计算所有训练样本的决策函数值
decision_values = oneclass_svm_model.decision_function(feature_arrays)
print(decision_values)
threshold = np.percentile(decision_values, 1)  # 假设允许5%的误拒率
print(threshold)
# # 4. 使用加载的模型预测
correct = 0
uncorrect = 0
y_val = []
y_val_pred = []
# predictions = oneclass_svm_model.decision_function(feature_arrays)
# for num in predictions:
#     y_val.append(0)
#     if num > -0.0005860172671331962:
#         y_val_pred.append(0)
#     else:
#         y_val_pred.append(1)

from sklearn.preprocessing import MultiLabelBinarizer

# y_val_flat = np.array(y_val).ravel()          # 真实标签
# y_val_pred_flat = np.array(y_val_pred).ravel()  # 预测标签
# val_accuracy = accuracy_score(y_val, y_val_pred)


# # 混淆矩阵
# cm = confusion_matrix(y_val_flat, y_val_pred_flat)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(2))
# disp.plot(include_values=True, cmap='Blues', xticks_rotation=90)
# plt.title(f'Confusion Matrix (Accuracy: {val_accuracy:.2f})')
# plt.tight_layout()
# plt.show()
# print(predictions.shape)


yolomodel = YOLO("../best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
yolomodel.to(device)
resmodel = models.resnet50(weights=None) #把权重文件下载到当前文件夹会更快
resmodel.load_state_dict(torch.load('./resnet50-19c8e357.pth'))  # 加载训练好的模型权重
resmodel.eval()
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
image_files = []
directory = '../未登记'
## 残差相关处理

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
        features = resmodel(image)
    # return features.numpy().flatten()
    return features

class ImageProcessor:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.output_path = os.path.join(input_folder)
        self.features = []
        # 预留处理接口（用户可自定义实现）
    
    def process_img(self):
        self.features = extract_features(self.input_folder)
    
        
processor = ImageProcessor("./box0.jpg")
for filename in os.listdir(directory):
    # 拼接完整路径，并检查是否为文件（而非子目录）
    filepath = os.path.join(directory, filename)
    img_test = cv2.imread(filepath)
    results = yolomodel.predict(img_test, conf=0.1, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) > 0:
        # 保存所有检测框
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = img_test[y1:y2, x1:x2]
            save_name = f"box0.jpg"
            cv2.imwrite("./box0.jpg", cropped)
            
            break

      

                # self.left_img.setPixmap(QPixmap("box0.jpg"))
    else:
        # 保存未检测图像
  
        save_path = "error.jpg"
        cv2.imwrite(str(save_path), img_test)
        
        break
            ## 残差网络提取特征值
    processor.process_img()

    ## pca标准化
    # 1. 标准化数据（推荐，尤其是特征量纲不一时）
    # 加载保存的参数
    components = np.load("pca_components.npy")  # 形状 (40, 1000)
    mean = np.load("pca_mean.npy")              # 形状 (1000,)
    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")
    processor.features = (processor.features - scaler_mean) / scaler_scale  # 标准化
            
            # 中心化 + 投影到主成分
    x_centered = (processor.features - mean)
    x_pca = np.dot(x_centered, components.T)  # 输出形状 (1, 40)
        
    ## svm分类

    result = oneclass_svm_model.decision_function(x_pca)
    y_val.append(1)
    if result >= -0.009:
        y_val_pred.append(0)
    else:
       y_val_pred.append(1)

from sklearn.preprocessing import MultiLabelBinarizer

# y_val_flat = np.array(y_val).ravel()          # 真实标签
# y_val_pred_flat = np.array(y_val_pred).ravel()  # 预测标签
# val_accuracy = accuracy_score(y_val, y_val_pred)
# 混淆矩阵
# cm = confusion_matrix(y_val_flat, y_val_pred_flat)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(2))
# disp.plot(include_values=True, cmap='Blues', xticks_rotation=90)
# plt.title(f'Confusion Matrix (Accuracy: {val_accuracy:.2f})')
# plt.tight_layout()
# plt.show()

from tqdm import tqdm
processor = ImageProcessor("./box1.jpg")
# 遍历所有子文件夹
for root, _, files in os.walk("../需实测图_登记_未登记混合"):
    # 计算当前子文件夹相对于输入根目录的路径
    rel_subfolder = os.path.relpath(root, "../login")
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
        img_path = os.path.join(root, f)
        img_test = cv2.imread(img_path)
        results = yolomodel.predict(img_test, conf=0.1, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            # 保存所有检测框
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = img_test[y1:y2, x1:x2]
                save_name = f"box1.jpg"
                cv2.imwrite("./box1.jpg", cropped)
                break

        

                    # self.left_img.setPixmap(QPixmap("box0.jpg"))
        else:
            # 保存未检测图像
    
            save_path = "error.jpg"
            cv2.imwrite(str(save_path), img_test)
            break
                ## 残差网络提取特征值
        processor.process_img()

        ## pca标准化
        # 1. 标准化数据（推荐，尤其是特征量纲不一时）
        # 加载保存的参数
        components = np.load("pca_components.npy")  # 形状 (40, 1000)
        mean = np.load("pca_mean.npy")              # 形状 (1000,)
        scaler_mean = np.load("scaler_mean.npy")
        scaler_scale = np.load("scaler_scale.npy")
        processor.features = (processor.features - scaler_mean) / scaler_scale  # 标准化
                
                # 中心化 + 投影到主成分
        x_centered = (processor.features - mean)
        x_pca = np.dot(x_centered, components.T)  # 输出形状 (1, 40)
            
        ## svm分类

        result = oneclass_svm_model.decision_function(x_pca)
        y_val.append(0)
        if result >= -0.009:
            y_val_pred.append(0)
        else:
            y_val_pred.append(1)
            print("...")

y_val_flat = np.array(y_val).ravel()          # 真实标签
y_val_pred_flat = np.array(y_val_pred).ravel()  # 预测标签
val_accuracy = accuracy_score(y_val, y_val_pred)
# 混淆矩阵
cm = confusion_matrix(y_val_flat, y_val_pred_flat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(2))
disp.plot(include_values=True, cmap='Blues', xticks_rotation=90)
plt.title(f'Confusion Matrix (Accuracy: {val_accuracy:.2f})')
plt.tight_layout()
plt.show()
for root, _, files in os.walk("../login"):
    # 计算当前子文件夹相对于输入根目录的路径
    rel_subfolder = os.path.relpath(root, "../login")
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
        img_path = os.path.join(root, f)
        img_test = cv2.imread(img_path)
        results = yolomodel.predict(img_test, conf=0.1, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            # 保存所有检测框
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = img_test[y1:y2, x1:x2]
                save_name = f"box1.jpg"
                cv2.imwrite("./box1.jpg", cropped)
                break

        

                    # self.left_img.setPixmap(QPixmap("box0.jpg"))
        else:
            # 保存未检测图像
    
            save_path = "error.jpg"
            cv2.imwrite(str(save_path), img_test)
            break
                ## 残差网络提取特征值
        processor.process_img()

        ## pca标准化
        # 1. 标准化数据（推荐，尤其是特征量纲不一时）
        # 加载保存的参数
        components = np.load("pca_components.npy")  # 形状 (40, 1000)
        mean = np.load("pca_mean.npy")              # 形状 (1000,)
        scaler_mean = np.load("scaler_mean.npy")
        scaler_scale = np.load("scaler_scale.npy")
        processor.features = (processor.features - scaler_mean) / scaler_scale  # 标准化
                
                # 中心化 + 投影到主成分
        x_centered = (processor.features - mean)
        x_pca = np.dot(x_centered, components.T)  # 输出形状 (1, 40)
            
        ## svm分类

        result = oneclass_svm_model.decision_function(x_pca)
        print(result)
        y_val.append(0)
        if result >= -0.009:
            y_val_pred.append(0)
        else:
            y_val_pred.append(1)
            print("...")

y_val_flat = np.array(y_val).ravel()          # 真实标签
y_val_pred_flat = np.array(y_val_pred).ravel()  # 预测标签
val_accuracy = accuracy_score(y_val, y_val_pred)
# 混淆矩阵
cm = confusion_matrix(y_val_flat, y_val_pred_flat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(2))
disp.plot(include_values=True, cmap='Blues', xticks_rotation=90)
plt.title(f'Confusion Matrix (Accuracy: {val_accuracy:.2f})')
plt.tight_layout()
plt.show()