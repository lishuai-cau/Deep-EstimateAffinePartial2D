import torch
from torch.utils.data import DataLoader
from utils.dataset import CocoDdataset
from model.homography import HomographyNet
from model.vgg import Vgg16_Homography
from model.resnet50 import resnet50
from model.mobilevit import mobile_vit_xx_small as create_mobile_vit
from model.swin_transformer import swin_base_patch4_window12_384 as create_swintiny
import argparse
import os
import numpy as np
import cv2
from utils.utils import save_correspondences_img
"""
    net = "HomographyNet" or "VGG" or "ResNet" or "MobileViT" or "SwinTransformer"
"""

net = "VGG"
# net = "ResNet"
# net = "MobileViT"
# net = "SwinTransformer"

def denorm_img(img):
    img = img * 127.5 + 127.5
    img = np.clip(img, 0, 255)
    return np.uint8(img)


def warp_pts(H, src_pts):
    dst_pts = np.zeros_like(src_pts)
    dst_pts[0] = (H[0,0] * src_pts[0]) + (H[0,1] * src_pts[1]) + H[0,2]
    dst_pts[1] = (H[1,0] * src_pts[0]) + (H[1,1] * src_pts[1]) + H[1,2]

    return dst_pts


def test(args):
    MODEL_SAVE_DIR = './checkpoints/'+net
    model_path = os.path.join(MODEL_SAVE_DIR, args.checkpoint)
    result_dir = './results/'+net
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if net == "VGG":
        model = Vgg16_Homography(num_classes=8)
    elif net == "HomographyNet":
        model = HomographyNet(num_classes=8)
    elif net == "ResNet":
        model = resnet50(pretrained=False, num_classes=8)
    elif net == "MobileViT":
        model = create_mobile_vit(num_classes=8)
    elif net == "SwinTransformer":
        model = create_swintiny(num_classes=8)
    else:
        raise ValueError('Unknown net type: {}'.format(net))

    state = torch.load(model_path)
    model.load_state_dict(state['state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
 
    TestingData = CocoDdataset(args.test_path)
    test_loader = DataLoader(TestingData, batch_size=1)

    print("start testing")
    with torch.no_grad():
        model.eval()
        error = np.zeros(len(TestingData))
        for i, batch_value in enumerate(test_loader):
            ori_images = batch_value[0].float()
            inputs = batch_value[1].float()
            pts1 = batch_value[2]
            target = batch_value[3].float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            outputs = model(inputs)
            outputs = outputs * 384
            target = target * 384
            # visual
            color_image = np.zeros((384, 384, 3), dtype=np.uint8)
            ir_image = np.zeros((384, 384, 3), dtype=np.uint8)
            ori_images = denorm_img(ori_images.numpy())
            for j in range(3):
                color_image[:, :, j] = ori_images[0][:,:,0]
                ir_image[:, :, j] = ori_images[0][:,:,1]
            pts1 = pts1[0].numpy()
            gt_h2p = target[0].numpy()
            pred_h2p = outputs[0].cpu().numpy()
            visual_file_name = ('%s' % i).zfill(4) + '.jpg'
            save_correspondences_img(color_image, ir_image, pred_h2p, result_dir, visual_file_name)
            error[i] = np.mean(np.sqrt(np.sum((gt_h2p - pred_h2p) ** 2.0/200.0, axis=-1)))
            print('Mean Corner Error: ', error[i])

        print('Mean Average Corner Error over the test set: ', np.mean(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="VGG_final.pth")
    parser.add_argument("--test_path", type=str, default="./data/testing/", help="path to test images")
    args = parser.parse_args()
    test(args)
