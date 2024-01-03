import os
import shutil
import numpy as  np
import cv2

"""
    net = "HomographyNet" or "VGG" or "ResNet"
"""


# 定义数据集文件夹路径
dataset_path = "./dataset"

# 定义划分后的文件夹路径
train_path = "./training"
val_path = "./validation"
test_path = "./testing"

# 定义训练集、验证集、测试集的比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 获取数据集中所有文件的路径
all_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        all_files.append(os.path.join(root, file))

# 计算训练集、验证集、测试集的数量
num_files = len(all_files)
num_train = int(num_files * train_ratio)
num_val = int(num_files * val_ratio)
num_test = num_files - num_train - num_val

# 划分数据集
train_files = all_files[:num_train]
val_files = all_files[num_train:num_train+num_val]
test_files = all_files[num_train+num_val:]

# 创建训练集、验证集、测试集文件夹
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 将文件复制到对应的文件夹中
for file in train_files:
    shutil.copy(file, train_path)

for file in val_files:
    shutil.copy(file, val_path)
    # ori_images, pts1, delta = np.load(file,allow_pickle=True)
    # out_images1 = ori_images[:, :, 0].copy()
    # out_images2 = ori_images[:, :, 1].copy()
    #
    # out_images = np.dstack((out_images1, out_images2))
    # np.save(os.path.join(val_path, os.path.basename(file)), (out_images, pts1, delta))
for file in test_files:
    shutil.copy(file, test_path)


# 打印训练集、验证集和测试集图像的数量
print("Number of training images: {}".format(len(train_files)))
print("Number of validation images: {}".format(len(val_files)))
print("Number of testing images: {}".format(len(test_files)))

