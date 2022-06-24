import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os


class FootDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.tranforms = transforms

        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name, ext = os.path.splitext(self.img_names[idx])
        img_path = os.path.join(self.img_dir, name + ext)
        pg_path = os.path.join(self.lbl_dir, name + '_pg' + ext)
        ft_path = os.path.join(self.lbl_dir, name + '_ft' + ext)

        # RGB
        img = cv2.imread(img_path)[:, :, ::-1]
        pg_mask = cv2.imread(pg_path, 0)
        ft_mask = cv2.imread(ft_path, 0)

        if self.tranforms:
            transformed = self.tranforms(image=img, masks=[pg_mask, ft_mask])
            img = transformed['image']
            pg_mask, ft_mask = transformed['masks']

        lbl =  np.stack([pg_mask, ft_mask], axis=2)

        # uint8 [0, 255] (h, w, c) to float [0., 1.] (c, h, w)
        to_tensor = ToTensor()
        img = to_tensor(img.copy())
        lbl = to_tensor(lbl.copy())

        return img, lbl


def get_dataloader(**kwargs):
    dataset = FootDataset(img_dir=kwargs['img_dir'], lbl_dir=kwargs['lbl_dir'])
    dataloader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=kwargs['shuffle'])
    return dataloader


if __name__ == '__main__':

    dataloader = get_dataloader(img_dir='../trainset/images', lbl_dir='../trainset/labels', batch_size=4, shuffle=True)
    images, labels = next(iter(dataloader))
    print(labels[0])