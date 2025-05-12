import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid  # 添加这行导入
import time  # 新增这行导入

# 内存分配


loaded = np.load(".npz", allow_pickle=True)
feature_arrays = loaded["data"]    # 特征向量对象数组
print(feature_arrays.shape)
classtype = np.repeat(np.arange(27), 10)  # 27个类别，每个类别10个样本

# 按类别分层划分训练+测试集（70%）和验证集（30%）
train_test_indices, val_indices = [], []
for i in range(27):
    class_indices = np.arange(i * 10, (i + 1) * 10)
    # np.random.shuffle(class_indices)  # 打乱每个类别的样本顺序
    train_test_indices.extend(class_indices[:7])  # 前7个为训练+测试
    val_indices.extend(class_indices[7:])  # 后3个为验证集

X_train_test, y_train_test = feature_arrays[train_test_indices], classtype[train_test_indices]
X_val, y_val = feature_arrays[val_indices], classtype[val_indices]

print("训练+测试集形状:", X_train_test.shape)
print("验证集形状:", X_val.shape)
from cuml.svm import SVC as cuSVC
## 标准化（pca已经标准化完成

# ## 模型训练

# # pip install --default-time=300 --extra-index-url=https://pypi.nvidia.com cuml-cu11

# # 定义参数搜索空间
# param_combinations = [
#     {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'},
#     {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'},
#     {'C': 10, 'gamma': 0.1, 'kernel': 'poly'}
# ]
# best_accuracy = 0
# best_params = None

# # # 手动搜索最优参数
# for params in param_combinations:
#     print('训练开始...\n')
#     model = cuSVC(probability=True, **params)
#     model.fit(X_train_test, y_train_test)
#     y_pred = model.predict(X_val)
#     acc = accuracy_score(y_val, y_pred)
#     print(f"参数: {params} -> 验证准确率: {acc:.4f}")
    
#     if acc > best_accuracy:
#         best_accuracy = acc
#         best_params = params

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

param_combinations = [
    # RBF 核扩展
    {'C': 0.001, 'gamma': 'scale', 'kernel': 'rbf'},
    {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf'},
    {'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'},
    {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'},
    {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'},
    {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'},
    
    # 线性核扩展
    {'C': 0.001, 'kernel': 'linear'},
    {'C': 0.01, 'kernel': 'linear'},
    {'C': 0.1, 'kernel': 'linear'},
    {'C': 1, 'kernel': 'linear'},
    {'C': 10, 'kernel': 'linear'},
    
    # 多项式核 (新增)
    {'C': 0.1, 'gamma': 'scale', 'kernel': 'poly', 'degree': 2, 'coef0': 0},
    {'C': 1, 'gamma': 'auto', 'kernel': 'poly', 'degree': 3, 'coef0': 1},
    {'C': 10, 'gamma': 0.1, 'kernel': 'poly', 'degree': 4, 'coef0': 0.5},
    
    # Sigmoid 核 (新增)
    {'C': 0.01, 'gamma': 'scale', 'kernel': 'sigmoid', 'coef0': 0},
    {'C': 1, 'gamma': 0.01, 'kernel': 'sigmoid', 'coef0': 1},
    {'C': 100, 'gamma': 0.001, 'kernel': 'sigmoid', 'coef0': -0.5}
]


best_accuracy = 0
best_params = None
maxcorrect = 0
#手动搜索最优参数（启用概率输出）
for params in param_combinations:
    print('训练开始...\n')
    # 使用 sklearn 的 SVC，并启用概率
    model = SVC(probability=True, **params)  # 关键修改点
    model.fit(X_train_test, y_train_test)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"参数: {params} -> 验证准确率: {acc:.4f}")
    
    
    y_val_pred = model.predict(X_val)
    probs = model.predict_proba(X_val) 
    # --- 计算特定指标 ---
    # 获取每个样本预测类别的对应概率
    pred_probs = probs[np.arange(len(probs)), y_val_pred]

    # 正确分类的样本掩码（True表示预测正确）
    correct_mask = (y_val == y_val_pred)

    # 1. 正确分类且概率 >0.2 的样本数
    correct_high_conf = np.sum(correct_mask & (pred_probs > 0.18))

    # 2. 所有概率 >0.2 的样本数（无论是否正确）
    total_high_conf = np.sum(pred_probs > 0.18)

    print(f"正确且概率>0.2的样本数: {correct_high_conf}")
    print(f"所有概率>0.2的样本数: {total_high_conf}")

    # if acc > (best_accuracy ) and (total_high_conf>maxcorrect):
    #     best_accuracy = acc
    #     maxcorrect = correct_high_conf
    #     best_params = params
    if  (total_high_conf>maxcorrect):
        best_accuracy = acc
        maxcorrect = correct_high_conf
        best_params = params

print("训练结束，最佳参数：", best_params, "\n",maxcorrect,best_accuracy)

import cloudpickle
final_model = SVC(probability=True, **best_params) 
final_model.fit(X_train_test, y_train_test)
# 保存模型
with open('sklsvc_model.pkl', 'wb') as f:
    cloudpickle.dump(final_model, f)

import cloudpickle

with open('sklsvc_model.pkl', 'rb') as f:
            loaded_model = cloudpickle.load(f)


# 使用最优参数训练最终模型（同样启用概率）

loaded_model.fit(X_train_test, y_train_test)





## 验证集评估与可视化
# 预测验证集
y_val_pred = loaded_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

probs = loaded_model.predict_proba(X_val) 
# --- 计算特定指标 ---
# 获取每个样本预测类别的对应概率
pred_probs = probs[np.arange(len(probs)), y_val_pred]

# 正确分类的样本掩码（True表示预测正确）
correct_mask = (y_val == y_val_pred)

# 1. 正确分类且概率 >0.2 的样本数
correct_high_conf = np.sum(correct_mask & (pred_probs > 0.18))

# 2. 所有概率 >0.2 的样本数（无论是否正确）
total_high_conf = np.sum(pred_probs > 0.18)

print(f"正确且概率>0.2的样本数: {correct_high_conf}")
print(f"所有概率>0.2的样本数: {total_high_conf}")

# 混淆矩阵
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(27))
disp.plot(include_values=True, cmap='Blues', xticks_rotation=90)
plt.title(f'Confusion Matrix (Accuracy: {val_accuracy:.2f})')
plt.tight_layout()
plt.show()

print(f"验证集准确率: {val_accuracy:.4f}")
# print("最优超参数:", best_params)





