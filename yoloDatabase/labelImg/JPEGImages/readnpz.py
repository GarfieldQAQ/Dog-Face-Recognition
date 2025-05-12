import numpy as np

# 加载npz文件
data = np.load('processed_data.npz')

# 查看所有保存的数组名称
print("包含的数组:", list(data.keys()))  # 输出如 ['features', 'filenames']

# 访问具体数组
features = data['data']
filenames = data['filenames']

# 验证数据
print("特征维度:", features.shape)      # 如 (1000, 2048)
print("文件名示例:", filenames[:])    # 如 ['1.jpg', '2.jpg', ...]