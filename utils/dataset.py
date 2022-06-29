import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.nn.functional import one_hot
import albumentations as A
import numpy as np
import os
import random


class FootDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, in_size, out_size, transform=None, n=768):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.in_size = in_size
        self.out_size = out_size
        self.n = n
        self.transform = transform

        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sample = random.choices(self.img_names, k=self.n)
        name, ext = os.path.splitext(sample[idx])
        img_path = os.path.join(self.img_dir, name + ext)
        pg_path = os.path.join(self.lbl_dir, name + '_pg' + ext)
        ft_path = os.path.join(self.lbl_dir, name + '_ft' + ext)

        # RGB
        img = cv2.imread(img_path)[:, :, ::-1]
        pg_mask = cv2.imread(pg_path, 0)
        ft_mask = cv2.imread(ft_path, 0)

        if self.transform:
            transformed = self.transform(image=img, masks=[pg_mask, ft_mask])
            img = transformed['image']
            pg_mask, ft_mask = transformed['masks']

        img = A.Resize(self.in_size, self.in_size)(image=img)['image']
        pg_mask, ft_mask = A.Resize(self.out_size, self.out_size)(image=img, masks=[pg_mask, ft_mask])['masks']

        lbl = np.zeros((self.out_size, self.out_size), dtype='int64')
        lbl[pg_mask == 255] = 1
        lbl[ft_mask == 255] = 2

        # uint8 [0, 255] (h, w, c) to float [0., 1.] (c, h, w)
        img = ToTensor()(img.copy())
        lbl = torch.from_numpy(lbl)
        # lbl = one_hot(lbl).permute(2, 0, 1).float()

        return img, lbl


def get_dataloader(**kwargs):
    dataset = FootDataset(
        img_dir=kwargs['img_dir'],
        lbl_dir=kwargs['lbl_dir'],
        in_size=kwargs['in_size'],
        out_size=kwargs['out_size'],
        transform=kwargs['transform']
    )
    dataloader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=kwargs['shuffle'])
    return dataloader


if __name__ == '__main__':

    dataloader = get_dataloader(img_dir='../devset/images', lbl_dir='../devset/labels', batch_size=4, in_size=224, out_size=216, transform=None, shuffle=True)
    images, labels = next(iter(dataloader))
    print(labels.shape)
