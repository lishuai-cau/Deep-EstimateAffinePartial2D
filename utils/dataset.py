import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2

# Create a customized dataset class in pytorch
class CocoDdataset(Dataset):
    def __init__(self, path, imsize=384):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = imsize

    def __getitem__(self, index):
        ori_images, pts1, delta = np.load(self.data[index], allow_pickle=True)

        #图像归一化
        ori_images = (ori_images.astype(float) - 127.5) / 127.5
        # ori_images =
        # print(pts1[0, 1])

        # input_patch = ori_images[:, pts1[0, 1]: pts1[1, 1], pts1[0, 0]: pts1[1, 0]]
        #网络固定的输入尺寸是(224*224)，所以这里要特别注意
        input_patch = np.transpose(ori_images, [2, 0, 1]) # torch [C,H,W]
        # 目标值归一化
        delta = delta.astype(float) / self.rho

        ori_images = torch.from_numpy(ori_images)
        input_patch = torch.from_numpy(input_patch)
        pts1 = torch.from_numpy(pts1)
        delta = torch.from_numpy(delta)
        return ori_images, input_patch, pts1, delta

    def __len__(self):
        return len(self.data)
