import torch.nn as nn
import torch.nn.functional as F


class Vgg16_Homography(nn.Module):
    def __init__(self,num_classes=1000):
        super(Vgg16_Homography, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            # 意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            # 意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            # 这样能够节省运算内存，不用多存储其他变量

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # (32-3+2)/1+1=32    32*32*64
            # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # (32-2)/2+1=16         16*16*64
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (8-2)/2+1=4      4*4*256
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (4-2)/2+1=2     2*2*512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (2-2)/2+1=1      1*1*512
        )



        self.fc = nn.Sequential(
            # y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            # nn.Liner(in_features,out_features,bias)
            # in_features:输入x的列数  输入数据:[batchsize,in_features]
            # out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            # bias: bool  默认为True
            # 线性变换不改变输入矩阵x的行数,仅改变列数


            nn.Linear(512*12*12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = out5.flatten(start_dim=1)
        output = self.fc(out6)
        return output


if __name__ == "__main__":
    from torchsummary import summary
    model =Vgg16_Homography(num_classes=8).cuda()
    print(Vgg16_Homography(num_classes=8))
    summary(model, (2, 384, 384))
