import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

'''
该文件由utils.py + train.py合并而成，对数据集加载+训练流程的表达很清晰。
对连哥哥get_loader函数进行重写，可以用作自己的数据集加载，比如把dset_name改为VOC等，
用自带的dataset = dsets.VOCDetection(...)会方便的多！！！
'''


def get_trainloader(dset_name, path, img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    if dset_name == "STL":
        dataset = dsets.STL10(root=path, split='train', transform=transform, download=True)
    elif dset_name == "CIFAR":
        dataset = dsets.CIFAR10(root=path, train=True, transform=transform, download=False)

    else:
        dataset = dsets.ImageFolder(root=path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.classes)

def get_testloader(dset_name, path, img_size, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    if dset_name == "STL":
        dataset = dsets.STL10(root=path, split='test', transform=transform, download=True)
    elif dset_name == "CIFAR":
        dataset = dsets.CIFAR10(root=path, train=False, transform=transform, download=True)
    else:
        dataset = dsets.ImageFolder(root=path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.classes)



def train(config):
    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)

    train_loader, num_class = utils.get_trainloader(config.dataset,
                                                    config.dataset_path,
                                                    config.img_size,
                                                    config.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = model.CNN(img_size=config.img_size, num_class=num_class).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=config.lr)

    min_loss = 999

    print("START TRAINING")
    for epoch in range(config.epoch):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % config.log_step == 0:
                if config.save_model_in_epoch:
                    torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                      % (epoch + 1, config.epoch, i + 1, len(train_loader), loss.item()))

        avg_epoch_loss = epoch_loss / len(train_loader)
        print('Epoch [%d/%d], Loss: %.4f'
              % (epoch + 1, config.epoch, avg_epoch_loss))
        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['STL', 'CIFAR', 'OWN'])
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--model_name', type=str, default='model.pth')

    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('-s', '--save_model_in_epoch', action='store_true')
    config = parser.parse_args()
    print(config)

    train(config)