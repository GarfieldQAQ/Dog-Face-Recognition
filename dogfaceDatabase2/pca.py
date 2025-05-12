import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 示例数据：假设有n个样本，每个样本1000维
n_samples = 80  # 样本数量
X = np.random.rand(n_samples, 1000)  # 生成随机数据
print(X.shape)

# 加载resnet处理的特征向量
loaded = np.load("processed_data.npz", allow_pickle=True)
feature_arrays = loaded["data"]    # 特征向量对象数组
print(feature_arrays.shape)

# 1. 标准化数据（推荐，尤其是特征量纲不一时）
scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
F_scaled = scaler.fit_transform(feature_arrays)

# 2. 初始化PCA，设置目标维度或保留方差比例
k = 80  # 目标维度
# 或自动选择保留95%方差的维度：
# pca = PCA(n_components=0.95)
pca = PCA(n_components=k)

# 3. 执行PCA降维
X_pca = pca.fit_transform(F_scaled)
components = pca.components_        # 主成分方向，形状 (40, 1000)
mean = pca.mean_                    # 训练数据的均值，形状 (1000,)
# 保存关键参数到当前文件夹
np.save("pca_components.npy", pca.components_)  # 保存主成分（协方差矩阵的特征向量）
np.save("pca_mean.npy", pca.mean_)             # 保存训练数据的均值
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)


# 加载保存的参数
saved_mean = np.load("pca_mean.npy")
saved_components = np.load("pca_components.npy")

# 检查与原始 PCA 模型是否一致
print("均值是否一致:", np.allclose(pca.mean_, saved_mean))
print("主成分是否一致:", np.allclose(pca.components_, saved_components))
# 输出结果
print("降维后的数据形状：", X_pca.shape)
print("累计方差解释率：", np.sum(pca.explained_variance_ratio_))





np.savez_compressed(
                "./",
                data=X_pca  
            )
