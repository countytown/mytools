import os
import glob
from os import listdir, getcwd
from os.path import join

train_img_path = "F:\\trainrotate\\train2017"   # 训练图片所在目录
train_txt_file = "F:\\trainrotate\\train2017.txt"   # train.txt路径

val_img_path = "F:\\trainrotate\\val2017"   # 验证图片所在目录
val_txt_file = "F:\\trainrotate\\val2017.txt"   # val.txt路径

test_img_path = "F:\\trainrotate\\test2017"  # 测试图片所在目录
test_txt_file = "F:\\trainrotate\\test2017.txt"  # test.txt路径




train_txt = open(train_txt_file,'a')  # 生成train.txt的内容
out_put_list = []
label_name = os.listdir(train_img_path)  # 获取训练图片的名称
for name in label_name:
    if name.endswith('jpg'):
        name = "F:\\trainrotate\\train2017\\" + name  # 指定训练图片的绝对路径
        out_put_list.append(name)
        train_txt.write(name+ '\n')


val_txt = open(val_txt_file,'a')
out_put_list = []
label_name = os.listdir(val_img_path)  # 获取验证图片的名称
for name in label_name:
    if name.endswith('jpg'):
        # if str(name).split('.')[0][0] == '2':
            name = "F:\\trainrotate\\val2017\\" + name  # 指定验证图片的绝对路径
            out_put_list.append(name)
            val_txt.write(name+ '\n')


test_txt = open(test_txt_file,'a')
out_put_list = []
label_name = os.listdir(test_img_path)  # 获取验证图片的名称
for name in label_name:
    if name.endswith('jpg'):
        # if str(name).split('.')[0][0] == '2':
            name = "F:\\trainrotate\\test2017\\" + name  # 指定测试图片的绝对路径
            out_put_list.append(name)
            test_txt.write(name+ '\n')
