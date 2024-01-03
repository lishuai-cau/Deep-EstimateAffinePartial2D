import torch
from torch import nn, optim
from utils.dataset import CocoDdataset
from utils.losscallback import LossHistory
from model.homography import HomographyNet
from model.vgg import Vgg16_Homography
from model.resnet50 import resnet50
from model.mobilevit import mobile_vit_xx_small as create_mobile_vit
from model.swin_transformer import swin_base_patch4_window12_384 as create_swintiny
from torch.utils.data import DataLoader
import argparse
import time
import os
import shutil

#自定义网络损失函数，用于计算loss
class MaxPointLoss(nn.Module):
    def __init__(self):
        super(MaxPointLoss, self).__init__()

    def forward(self, input, target):
        loss = torch.max((input - target)** 2)
        return loss


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
"""
    net = "HomographyNet" or "VGG" or "ResNet" or "MobileViT" or "SwinTransformer"
"""
net = "VGG"
# net = "ResNet"
# net = "MobileViT"
# net = "SwinTransformer"
# net = "HomographyNet"



def train(args):
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

    MODEL_SAVE_DIR = 'checkpoints/'+net
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    #删除runs文件夹
    if os.path.exists('./runs/'+net):
        shutil.rmtree('./runs/'+net)
    loss_history = LossHistory(log_dir='./runs/'+net, model=model, input_shape=(384, 384))
    TrainingData = CocoDdataset(args.train_path)
    ValidationData = CocoDdataset(args.val_path)
    print('Found totally {} training files and {} validation files'.format(len(TrainingData), len(ValidationData)))
    train_loader = DataLoader(TrainingData, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ValidationData, batch_size=args.batch_size, num_workers=4)

    if torch.cuda.is_available():
        model = model.cuda()

    MSEcriterion = nn.MSELoss()  #均方根误差
    Maxcriterion = MaxPointLoss()  # 最大点误差
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # decrease the learning rate after every 1/3 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs / 3), gamma=0.1)

    print("start training")
    glob_iter = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Training
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        for i, batch_value in enumerate(train_loader):
            # save model
            # if (glob_iter %  (args.batch_size*10) == 0 and glob_iter != 0):
            if epoch% 500 == 0 and epoch != 0:
                filename = net + '_iter_' + str(epoch) + '.pth'
                model_save_path = os.path.join(MODEL_SAVE_DIR, filename)
                state = {'epoch': args.epochs, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, model_save_path)

            if epoch == args.epochs-1:
                filename = net + '_final.pth'
                model_save_path = os.path.join(MODEL_SAVE_DIR, filename)
                state = {'epoch': args.epochs, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, model_save_path)

            ori_images = batch_value[0].float()
            inputs = batch_value[1].float()
            pts1 = batch_value[2]
            target = batch_value[3].float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)
            lossmse = MSEcriterion(outputs, target)
            lossmax = Maxcriterion(outputs, target)
            lossmax.backward()
            optimizer.step()
            train_loss += lossmax.item()
            train_mse += lossmse.item()
            if (i + 1) % 200 == 0 or (i+1) == len(train_loader):
                print("Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] MaxPoint Error: {:.4f} Mean Squared Error: {:.4f} lr={:.6f}".format(
                    epoch+1, args.epochs, i+1, len(train_loader), train_loss / 200, train_mse / 200,scheduler.get_last_lr()[0]))
                total_loss = train_loss
                train_loss = 0.0
            glob_iter += 1
            # print("glob_iter: ", glob_iter)
        scheduler.step()

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            for i, batch_value in enumerate(val_loader):
                ori_images = batch_value[0].float()
                inputs = batch_value[1].float()
                pts1 = batch_value[2]
                target = batch_value[3].float()
                if torch.cuda.is_available():
                    inputs, target = inputs.cuda(), target.cuda()
                outputs = model(inputs)

                lossmax = Maxcriterion(outputs, target)
                lossmse = MSEcriterion(outputs, target)
                val_loss += lossmax.item()
                val_mse += lossmse.item()
            print("Validation: Epoch[{:0>3}/{:0>3}] MaxPoint Error: {:.4f} Mean Squared Error:{:.4f}, epoch time: {:.1f}s".format(
                epoch + 1, args.epochs, val_loss / len(val_loader), val_mse / len(val_loader),time.time() - epoch_start))

        loss_history.append_loss(epoch + 1, total_loss / 200, val_loss / len(val_loader),train_mse / 200, val_mse / len(val_loader))
    elapsed_time = time.time() - t0
    print("Finished Training in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))


if __name__ == "__main__":
    train_path = './data/training/'
    val_path = './data/validation/'

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")

    parser.add_argument("--train_path", type=str, default=train_path, help="path to training imgs")
    parser.add_argument("--val_path", type=str, default=val_path, help="path to validation imgs")
    args = parser.parse_args()
    train(args)
