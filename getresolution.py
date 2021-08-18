from PIL import Image
import os

imgdir = r'F:\img_folder'
txtdir = r'F:\txtfolder'  #origin labels, xywh are not normalized
newtxtdir = r'F:\new_txtfolder' #folder to save txt files after normalized
for file in os.listdir(imgdir):
    one_file_path = os.path.join(imgdir,file)
    img_file_name = file.split('.')[0]

    im = Image.open(one_file_path)#返回一个Image对象
    # print('宽：%d,高：%d'%(im.size[0],im.size[1]))
    width = im.size[0]
    height = im.size[1]
    for txtfile in os.listdir(txtdir):
        save_dir = newtxtdir
        txt_file_name = txtfile.split('.')[0]
        if txt_file_name == img_file_name:
            one_txt_path = os.path.join(txtdir, txtfile)
            for line in open(one_txt_path):
                if line == '\n':
                    print('发现换行')
                    pass
                else:
                    one_list = line.split(' ')
                    print(one_list,'===')
                    cls = one_list[0]
                    x = float(one_list[1])/width
                    x = round(x,6)
                    y = float(one_list[2]) / height
                    y = round(y, 6)
                    w = float(one_list[3]) / width
                    w = round(w, 6)
                    h = float(one_list[4]) / height
                    h = round(h, 6)
                    '''degree process  if only cls xywh, delete one_lisr[5]'''
                    degree_360 = round(float(one_list[5][:-1])*180/3.1415)

                    yolo_longside_str = cls+" "+str(x)+" "+ str(y) + " "+str(w) +" "+str(h)+" " +str(degree_360)+'\n'


                save_path = os.path.join(save_dir,txtfile)
                with open(save_path, "a") as f:
                    try:
                        f.write(yolo_longside_str)
                    except:
                        print('write failed')