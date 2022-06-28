import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import ToTensor

from models.unet import UNet


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--weights', type=str, default='weights/ckpt.pth')
    ap.add_argument('--input', type=str, default=None)
    ap.add_argument('--output', type=str, default='output')

    args = vars(ap.parse_args())

    return args


def main(args):

    if not os.path.isdir(args['output']):
        os.makedirs(args['output'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(3).to(device)

    net.load_state_dict(torch.load(args['weights'], map_location=device))

    img = cv2.imread(args['input'])
    scale = 350/max(img.shape)
    W, H = int(img.shape[1] * scale), int(img.shape[0] * scale)
    img = cv2.resize(img, (W, H))
    x = ToTensor()(img[:, :, ::-1].copy()).unsqueeze(0)

    y = np.where(net(x).squeeze() > 0.5, 1, 0).astype('uint8')
    pg_mask, ft_mask = y[0], y[1]

    pg_mask = cv2.resize(pg_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    ft_mask = cv2.resize(ft_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    collage = np.concatenate([pg_mask, ft_mask], axis=1)
    plt.imshow(collage)
    plt.show()

    return pg_mask, ft_mask


if __name__ == '__main__':

    args = parse_opt()

    main(args)