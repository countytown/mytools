import os
import glob
from PIL import Image
import shutil

'''①全部标签文件 ②部分图片文件 用于将提取图片文件对应的标签文件'''

root_dir = r"C:\Users\LH\Desktop\test_copy_label"
source_img = r"C:\Users\LH\Desktop\test_copy_label\imgs\voc2028_blured"
old_lable_path = r"C:\Users\LH\Desktop\test_copy_label\_old_labels\voc2028"

img_name_list = [] #img_name_list，包括文件后缀格式；
label_name_list = []
#imgname1 = [] #imgname1指里面的文件名称，不包括文件后缀格式

#通过glob.glob来获取第一个文件夹下，所有'.jpg'文件
imageList1 = glob.glob(os.path.join(source_img, '*.jpg'))
for item in imageList1:
    temp = os.path.basename(item)
    item_name = temp.split(".")[0]
    img_name_list.append(item_name) #现在image1是所有图片的无后缀名称

labelList1 = glob.glob(os.path.join(old_lable_path, '*.txt'))
for item in labelList1:
    temp = os.path.basename(item)
    item_name = temp.split(".")[0]
    label_name_list.append(item_name) #现在label1是所有标签的无后缀名称

# print(label_name_list)
# print(img_name_list)
label_src = root_dir + "\\"+"_old_labels\\voc2028"+"\\"
label_dir = root_dir + "\\"+"_new_labels"+"\\"

for label_item in label_name_list:
    for  img_item in img_name_list:
        if label_item == img_item:
            label_src_file = label_src + label_item +".txt" #单个待复制label的源路径
            label_dir_file = label_dir + label_item +".txt" #单个待复制label的目的路径
            #print(label_src_file)
            shutil.copyfile(label_src_file, label_dir_file)
        else:
            print("无此txt文件")



