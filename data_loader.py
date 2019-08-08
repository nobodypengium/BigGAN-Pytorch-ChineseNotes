import torch
import torchvision.datasets as dsets
from torchvision import transforms


class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160)) #这里定死160了，所以貌似只能128，从中间截图
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes=['church_outdoor_train','classroom_train']):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=classes, transform=transforms)
        return dataset
    
    def load_imagenet(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/imagenet', transform=transforms) #读取图片子文件夹类的文件结构
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        # dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset

    def load_off(self):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'imagenet':
            dataset = self.load_imagenet()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'off':
            dataset = self.load_off()

        print('dataset',len(dataset))
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2, #多少个子进程加载数据
                                              drop_last=True) #不够batch_size要不要丢掉最后一撮数据
        return loader

