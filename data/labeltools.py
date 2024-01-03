#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import  numpy as np
global ori_img,ir_img,color_img,point,count,filename
import  os
from numpy.linalg import inv
import platform
from PIL import Image
count = 0
point ={}
new_path = './dataset'

def padding_image(img):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转变成Image
    image = Image.fromarray(np.uint8(frame))
    width, height = image.size
    target_width = 384
    target_height = 384
    # 计算要添加到左侧、右侧、顶部和底部的灰色条的大小
    left_padding = (target_width - width) // 2
    right_padding = target_width - width - left_padding
    top_padding = (target_height - height) // 2
    bottom_padding = target_height - height - top_padding
    # 创建一个新的图像对象，大小为目标宽度和高度，背景色为灰色
    new_image = Image.new("RGB", (target_width, target_height), (128, 128, 128))
    # 计算原始图像粘贴的位置
    paste_position = (left_padding, top_padding)
    # 粘贴原始图像到新图像
    new_image.paste(image, paste_position)
    opencv_image = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)
    return opencv_image

def img_perspect_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)

def on_mouse(event,x,y,flags,param):
    global color_img,ir_img,ori_img,point,count,filename
    # img=ori_img.copy()
    if event==cv2.EVENT_LBUTTONDOWN:#左键点击
        count = count+1
        if count %2 ==0:
            point[count-1]=(x,y)
            cv2.circle(ori_img,point[count-1],10,(0,255,0),5)
        else:
            point[count-1]=(x,y)
            cv2.circle(ori_img, point[count-1], 10, (0, 255, 0), 5)
        cv2.imshow(filename,ori_img)


    elif event==cv2.EVENT_LBUTTONUP:#左键释放
        if count %2 ==0:
            print("point1 to point2",point[count-2],point[count-1])
            cv2.line(ori_img,point[count-2],point[count-1],(0,0,255),3)
            cv2.imshow(filename, ori_img)
        if count % 4 == 0:

            ir_two_points = [(point[count - 3][0]-384,point[count - 3][1]),(point[count - 1][0]-384,point[count - 1][1])]     #红外图
            rgb_two_points = [point[count-4],point[count-2]]     #原图
            H = cv2.estimateAffinePartial2D(np.float32(ir_two_points), np.float32(rgb_two_points))
            # print(H)
            # H_inverse = inv(H)    #求逆矩阵
            # warped_image为红外图像经过H矩阵变换后的图像
            warped_image = cv2.warpAffine(ir_img, np.float32(H[0]), (color_img.shape[1], color_img.shape[0]))
            combine = cv2.addWeighted(color_img,0.4,warped_image,0.8,0)
            #数据集格式是rgb ir warped_image
            print(color_img.shape[2],ir_img.shape[2],warped_image.shape[2])

            training_image =  np.zeros((384,384, 2), dtype=np.uint8)
            training_image[:, :, 0] = color_img[:, :, 0]
            training_image[:, :, 1] = ir_img[:, :, 0]
            # training_image[:, :, 2] = warped_image[:, :, 0]
            print(training_image.shape[2])
            H_points= np.ravel(np.hstack((np.array(ir_two_points), np.array(rgb_two_points))))
            datum = (training_image, np.array(ir_two_points), np.array(H_points))
            # print(datum)
            cv2.imshow("perspect",combine)
            save_flag = cv2.waitKey(0)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            if save_flag == ord('s'):
                np.save(new_path + '/' + ('%s' % int(filename.split('.')[0])).zfill(6), datum)
                print("rgb_two_points",rgb_two_points)
                print("ir_points",ir_two_points)
                print("save image %s" % filename)
            else:
                print("not save image %s" % filename)

def read_path(file_pathname,mode = 'gray',init_count = 1):
    color_filepath = file_pathname + '/color'
    ir_filepath = file_pathname + '/ir'

    #获取文件夹当中的文件个数
    global ori_img, ir_img, color_img, filename
    total_num_color = len(os.listdir(color_filepath))
    total_num_ir = len(os.listdir(ir_filepath))
    if total_num_color != total_num_ir:
        print("the number of color image is not equal to ir image")
        return -1
    print("total image is %d " % total_num_color)
    for filename in os.listdir(color_filepath):


        if int(filename.split('.')[0]) < init_count:
            continue


        if mode == 'gray':
            color_img = cv2.imread(color_filepath+'/'+filename)
            ir_img = cv2.imread(ir_filepath+'/'+filename)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
            color_img = padding_image(color_img)
            ir_img = padding_image(ir_img)

        elif mode == 'color':
            color_img = cv2.imread(color_filepath + '/' + filename)
            ir_img = cv2.imread(ir_filepath + '/' + filename)
            color_img = padding_image(color_img)
            ir_img = padding_image(ir_img)

        else:
            print("please set mode == gray or mode == color")
            return -1
        cv2.namedWindow(filename)
        cv2.setMouseCallback(filename,on_mouse)
        ori_img = np.hstack((color_img,ir_img))
        cv2.imshow(filename,ori_img)
        if platform.system().lower() == 'windows':
            print("current platform is windows")
            cv2.waitKey(0)
        elif platform.system().lower() == 'linux':
            print("current platform is linux")
            while cv2.waitKey(100) != 27:
                if cv2.getWindowProperty(filename, cv2.WND_PROP_VISIBLE) <= 0:
                    break


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    read_path("./Image",mode='gray',init_count=2911)

    #数据集载入测试
    # ori_images, pts1, delta  =  np.load("./training/000001.npy",allow_pickle=True)
    # print(ori_images.shape)
    # color_image = np.zeros((384, 384, 3), dtype=np.uint8)
    # ir_image = np.zeros((384, 384, 3), dtype=np.uint8)
    # for i in range(3):
    #     color_image[:, :, i] = ori_images[:, :, 0]
    #     ir_image[:, :, i] = ori_images[:, :, 1]
    # cv2.imshow("color_image", color_image)
    # cv2.imshow("ir_image", ir_image)
    # cv2.waitKey(0)


