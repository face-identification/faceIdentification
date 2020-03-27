import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image


class faceDataset(Dataset):
    def __init__(self, id_dir, resize_height=64, resize_width=64):
        '''
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为图像高，
        :param resize_width  为图像宽
        '''
        # 所有图片的绝对路径
        self.imgs = []
        self.labels = []
        for id in range(len(id_dir)):
            imgs = os.listdir('./faces/'+id_dir[id])
            for k in imgs:
                self.imgs.append(os.path.join('./faces/'+id_dir[id], k))
                self.labels.append(id)
            # 相关预处理的初始化
            #  self.transforms=transform
            self.transforms = True
            self.transform = transforms.Compose([
                transforms.Resize((resize_height,resize_width)),
                transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
            ])

    def __getitem__(self, i):
        img_path = self.imgs[i]
        label = self.labels[i]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transform(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data,label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    image_dir = ['1','2']  # 该文件夹下面直接是图像，与原始文件不一样，原始文件是有人名的二级目录

    epoch_num = 1  # 总样本循环次数 反对
    batch_size = 2 # 训练时的一组数据的大小
    train_data_nums = 20
    max_iterate = int((train_data_nums + batch_size - 1) / batch_size * epoch_num)  # 总迭代次数
    train_data = faceDataset(id_dir=image_dir)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    # [1]使用epoch方法迭代，LfwDataset的参数repeat=1
    for epoch in range(epoch_num):
        for batch_image,label in train_loader:
            image = batch_image[0, :]
            image = image.numpy()  #
            # plt.imshow(image)
            # plt.show()
            print("batch_image.shape:{}".format(batch_image.shape))
            print(label)