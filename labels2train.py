import os
import glob
from os import listdir, getcwd
from os.path import join

train_img_path = "F:\sports\coco\\train2017"   # 训练图片所在目录
train_txt_file = "F:\sports\coco\\train.txt"   # train.txt路径

val_img_path = "F:\sports\coco\\val2017"  # 验证图片所在目录
val_txt_file = "F:\sports\coco\\val.txt"  # val.txt路径

train_txt = open(train_txt_file,'a')  # 生成train.txt的内容
out_put_list = []
label_name = os.listdir(train_img_path)  # 获取训练图片的名称
for name in label_name:
    if name.endswith('jpg'):
        name = "F:\sports\coco\images\\" + name  # 指定训练图片的绝对路径
        out_put_list.append(name)
        train_txt.write(name+ '\n')


val_txt = open(val_txt_file,'a')
out_put_list = []
label_name = os.listdir(val_img_path)  # 获取验证图片的名称
for name in label_name:
    if name.endswith('jpg'):
        name = "F:\sports\coco\images\\" + name  # 指定验证图片的绝对路径
        out_put_list.append(name)
        val_txt.write(name+ '\n')