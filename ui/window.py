# -*- coding: utf-8 -*-

# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2
from PIL import Image
import os.path as osp
import cloudpickle
import torchvision.transforms as transforms
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

import torchvision.models as models
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FILE = Path(__file__).resolve()



# 调试窗口类
class DebugWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Debug信息")
        self.setGeometry(100, 100, 400, 300)  # 窗口位置和大小
        
        # 文本显示区域
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        
        # 简单布局
        layout = QVBoxLayout()
        layout.addWidget(self.text_area)
        self.setLayout(layout)

    def log(self, message):
        """添加日志"""
        self.text_area.append(message)


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
        features = mainWindow.resmodel(image)
    # return features.numpy().flatten()
    return features

def sift_similarity(img1, img2, N=100):
    # 灰度处理
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建 SIFT 检测器
    sift = cv2.SIFT_create()

    # 提取所有关键点
    kp1_all = sift.detect(img1, None)
    kp2_all = sift.detect(img2, None)

    # 按 response 取前 N 个“好点”
    kp1 = sorted(kp1_all, key=lambda kp: kp.response, reverse=True)[:N]
    kp2 = sorted(kp2_all, key=lambda kp: kp.response, reverse=True)[:N]

    # 计算描述子
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)

    # 用 FLANN 匹配器 + Lowe ratio
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe ratio 筛选
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # 如果匹配太少就直接返回 0 相似度
    if len(good) < 4:
        return 0.0

    # 用 RANSAC 计算单应矩阵，筛掉离群点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = mask.ravel().sum()

    # 相似度 = RANSAC 内点数 / Lowe 匹配数（或者改成 / N）
    similarity = inliers / len(good)
    return similarity
class ImageProcessor:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.output_path = os.path.join(input_folder)
        self.features = []
        # 预留处理接口（用户可自定义实现）
    
    def process_img(self):
        self.features = extract_features(self.input_folder)
    
        
processor = ImageProcessor("./box0.jpg")
def cosine_similarity_np(a, b):
        
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):

        # 工作目录获取
        self.workpath = os.getcwd()    #获取当前工作目录
        print(self.workpath)
        aim_workpath = self.workpath.split("ui",1)[0]
        self.workpath = aim_workpath
        
        # 初始化界面
        super().__init__()
        self.setWindowTitle('Target detection system')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        with open('sklsvc_model.pkl', 'rb') as f:
            loaded_model = cloudpickle.load(f)
        self.svmmodel = loaded_model
        with open('oneclass_svm_model.pkl', 'rb') as f:
            oneclass_svm_model = cloudpickle.load(f)
        self.oneclasssvmmodel = oneclass_svm_model
        yolomodel = YOLO("best.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        yolomodel.to(device)
        self.yolomodel = yolomodel
        # self.model = self.model_load(weights="runs/train/exp_yolov5s/weights/best.pt",
        #                              device=self.device)  # todo 指明模型加载的位置的设备
        # 加载预训练的ResNet50模型
        resmodel = models.resnet50(weights=None) #把权重文件下载到当前文件夹会更快
        resmodel.load_state_dict(torch.load('./resnet50-19c8e357.pth'))  # 加载训练好的模型权重
        resmodel.eval()
        self.resmodel = resmodel
        self.dict ={0: '宋总的狗-中黄（中华田园犬）', 1: '宋总的狗-二百（吉娃娃）', 2: '宋总的狗-多比（迷你宾莎犬）', 3: '宋总的狗-大乖(迷你宾莎犬)', 4: '宋总的狗-大黄（中华田园犬）', 5: '宋总的狗-小白（法国斗牛犬）', 6: '宋总的狗-皮特（迷你宾莎犬）', 7: '宋总的狗-豆豆（迷你宾莎犬）', 8: '宠合里-博美', 9: '宠合里-斑点犬', 10: '宠合里-柯基犬', 11: '宠合里-柴犬', 12: '宠合里-比格犬', 13: '宠合里-沙皮狗', 14: '宠合里-泰迪犬', 15: '宠合里-红哈士奇', 16: '宠合里-约克夏', 17: '宠合里-腊肠', 18: '宠合里-萨摩耶', 19: '宠合里-贵宾犬', 20: '宠合里-边牧', 21: '宠合里-金毛', 22: '宠合里-黑哈士奇', 23: '小高的百万-拉布拉多', 24: '罗晓琦的拉布拉多', 25: '邹娇的糯米-边牧', 26: '郎诚的克拉米-迷你宾莎犬'}
        self.initUI()
        self.reset_vid()
        # 创建调试窗口并显示
        self.debug_win = DebugWindow()
        self.debug_win.show()  # 同时显示调试窗口
        
        # 示例：直接添加一条日志
        self.debug_win.log("程序启动成功")

        

    '''
    ***模型初始化***
    '''
    # @torch.no_grad()
    def model_load(self):
    # 加载并预测
        with open('cusvc_model.pkl', 'rb') as f:
            model = cloudpickle.load(f)
        print("模型加载完成!")
        return model

    # '''
    # ***界面初始化***
    # '''
    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("Photo Predict")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("Upload Photo")
        det_img_button = QPushButton("Start!")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)


        # todo 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('Doggy System\n\n ')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/qq.png'))
        about_img.setAlignment(Qt.AlignCenter)

        

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, 'img predict')
       

    # '''
    # ***上传图片***
    # '''
    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
            self.uploadimg = im0

    # '''
    # ***检测图片***
    # '''


    

    def detect_img(self):
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        if source == "":
            QMessageBox.warning(self, "Please Upload", "Upload First")
        else:
            source = str(source)
            img = cv2.imread(source)
            results = self.yolomodel.predict(img, conf=0.1, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
        # 处理检测结果
        if len(boxes) > 0:
            # 保存所有检测框
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = img[y1:y2, x1:x2]
                save_name = f"box0.jpg"
                cv2.imwrite("./box0.jpg", cropped)
                break

            self.dogflag = 1

            # self.left_img.setPixmap(QPixmap("box0.jpg"))
        else:
            # 保存未检测图像
            self.dogflag = 0
            save_path = "error.jpg"
            cv2.imwrite(str(save_path), img)
            self.right_img.setPixmap(QPixmap("err.jpg"))
            print("未采集到狗头")
            self.debug_win.log("未采集到狗头")
            return

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
        

        ## oneclass分类
        y_val_pred = self.oneclasssvmmodel.decision_function(x_pca)
        if y_val_pred > -0.0005860172671331962:

            ## svm分类
            y_val_pred = self.svmmodel.predict(x_pca)
            predicted_class = y_val_pred.item()
            self.debug_win.log("归类为：")
            self.debug_win.log(str(predicted_class))
            self.debug_win.log(str(self.dict[predicted_class]))
            path = str("../login/"+str(self.dict[predicted_class])+"/1.jpg")
            self.debug_win.log(path)

            resultimg = cv2.imread(path)
            resize_scale = self.output_size / resultimg.shape[0]
            resultimg = cv2.resize(resultimg, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("./show_result.jpg", resultimg)
            self.right_img.setPixmap(QPixmap("./show_result.jpg"))
        else:
            print(y_val_pred)
            print("不在库中...")
            self.debug_win.log("不在库中...")
            self.right_img.setPixmap(QPixmap("err.jpg"))
            return

        

       

    # 视频检测，逻辑基本一致，有两个功能，分别是检测摄像头的功能和检测视频文件的功能，先做检测摄像头的功能。
    # '''
    # ### 界面关闭事件 ### 
    # '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    # '''
    # ### 界面重置事件 ### 
    # '''

    def reset_vid(self):
        print("404\n")
       

    # '''
    # ### 视频重置事件 ### 
    # '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
