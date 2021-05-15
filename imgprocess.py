'''
transform for dataset
在进行resize，corp等操作前要确保图片被转换为了PIL格式，在处理完后还要转回Tensor
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# global transform
transform = transforms.Compose([
    transforms.ToPILImage(),  # 下面的函数输入都是PIL
    transforms.Resize(size=(224, 224)),  # 根据网络需要修改
    transforms.RandomRotation(90),  # 旋转，增加训练样本数量
    transforms.RandomHorizontalFlip(0.5),  # 水平翻转，增加训练样本数量
    transforms.RandomVerticalFlip(0.5),  # 垂直翻转，增加训练样本数量
    transforms.ToTensor(),
    normalize
])

'''该collate_fn是在加载自制VOC数据集时，遇到了以下问题所写的:
在以batch>1加载时，由于每张图片对应的GT label数量不一致，在堆叠图片为batch时报大小不一致，
所以在网上找了该代码修改，先将tensor大小一致化，小的tensor用-1？补齐，再返回一个batch,解决了报错。
'''


def collate_fn(data):
    imgs_list, boxes_list = zip(*data)
    imgs_list = [torch.tensor(img) for img in imgs_list]
    boxes_list = [torch.tensor(box) for box in boxes_list]

    assert len(imgs_list) == len(boxes_list)
    batch_size = len(boxes_list)
    pad_imgs_list = []
    pad_boxes_list = []
    pad_classes_list = []

    h_list = [int(s.shape[1]) for s in imgs_list]  # 每张图像的高度
    w_list = [int(s.shape[2]) for s in imgs_list]  # 每张图像的宽度
    # print(h_list)
    max_h = np.array(h_list).max()
    max_w = np.array(w_list).max()
    for i in range(batch_size):  # 对每张图片进行padding，以达到相同的形状
        img = imgs_list[i]
        pad_imgs_list.append(
            torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.))

    max_num = 0
    '''
    boxes_list包含了n张图片对应的目标box+cls_id， n取决于batch包含的图片数目
    '''
    for i in range(batch_size):
        n = boxes_list[i].shape[0]  # 第i张图片的有n个box+cls_id
        # print(boxes_list,"nn")
        # print(n)
        if n > max_num: max_num = n  # 为了获取最多有多少个obj
    for i in range(batch_size):
        pad_boxes_list.append(
            torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
    #
    #
    batch_boxes = torch.stack(pad_boxes_list)
    # batch_classes = torch.stack(pad_classes_list)
    batch_imgs = torch.stack(pad_imgs_list)

    return batch_imgs, batch_boxes


'''调用方法'''
data_loader = VocDataset(root_dir='path/NWPU_VHR_10_dataset/VOCdevkit', image_set='test', transform=transform)
train_loader = torch.utils.data.DataLoader(data_loader, batch_size=2, shuffle=False, collate_fn=collate_fn)

'''特征图/数据集图片PLT显示方法'''
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from PIL import Image

def show_img(ori_img, f1, f2, name='P'):
    for i in range(1, 3):
        axe1 = plt.subplot(1, 3, 1)  # 一行三列的图片
    axe1.set_title(name)  # 设置每张图的title
    plt.imshow(ori_img)
    axe2 = plt.subplot(1, 3, 2)
    axe2.set_title('f1')
    plt.imshow(f1)
    axe3 = plt.subplot(1, 3, 3)
    axe3.set_title('f2')
    plt.imshow(f2)
    plt.show()  # 最终显示


'''
显示刚从数据集中加载出来的图片

'''

resize_2tensor = transforms.Compose([
    transforms.CenterCrop((128, 128)),  # 只能对PIL图片进行裁剪
    transforms.ToTensor(),
]
)
for imgfile in os.listdir('img_dir'):
    imgpath = os.path.join('.\data\drone', imgfile)
    plt_image = Image.open(imgpath)
    plt_image.show()  # 可直接显示原图
    tensor_img = resize_2tensor(plt_image)  # 对原来的plt img进行resize+totensor操作后，方便后续送入模型
    dtensor_img_3d = tensor_img.unsqueeze(0).cuda()  # 送入模型需要添加一维batch!! [batch,channel,height,width]
    # print(dtensor_img_3d.shape, '3D')
    logit, _ = cnn(dtensor_img_3d)  # 送入模型

def expand_one2three_channels(map):
    np_onec = map[0][0].numpy()
    image = np.expand_dims(np_onec, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    cv2.imshow("3channel",image)
    cv2.waitKey(0)
