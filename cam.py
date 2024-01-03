import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils.utils import GradCAM, show_cam_on_image, center_crop_img
from model.vgg import Vgg16_Homography

net = "VGG"
MODEL_SAVE_DIR = './checkpoints/VGG/VGG_final.pth'



def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]
    model = Vgg16_Homography(num_classes=8)
    state = torch.load(MODEL_SAVE_DIR)
    model.load_state_dict(state['state_dict'])
    target_layers = [model.layer5]
    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # # load image
    # img_path = "both.png"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path).convert('RGB')
    # img = np.array(img, dtype=np.uint8)
    # # img = center_crop_img(img, 224)
    #
    # # [C, H, W]
    # img_tensor = data_transform(img)
    # # expand batch dimension
    # [C, H, W] -> [N, C, H, W]


    ori_images, pts1, delta = np.load("test.npy", allow_pickle=True)
    color_image = np.zeros((384, 384, 3), dtype=np.uint8)
    for j in range(3):
        color_image[:, :, j] = ori_images[:, :, 0]
    ori_images = (ori_images.astype(float) - 127.5) / 127.5
    input_patch = np.transpose(ori_images, [2, 0, 1])  # torch [C,H,W]
    input_tensor = torch.from_numpy(input_patch)
    input_tensor = input_tensor.float()
    input_tensor = input_tensor.unsqueeze(0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 1  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    #
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(color_image.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
