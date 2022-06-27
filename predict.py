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
    net = UNet(3, 2).to(device)

    net.load_state_dict(torch.load(args['weights'], map_location=device))

    img = cv2.imread(args['input'])
    img = cv2.resize(img, (512, 512))
    img = ToTensor()(img[:, :, ::-1].copy()).unsqueeze(0)

    y_hat = np.where(net(img).squeeze() > 0.5, 1, 0).astype('uint8')
    pg_mask, ft_mask = y_hat[0], y_hat[1]
    collage = np.concatenate([pg_mask, ft_mask], axis=1)

    # cv2.imwrite(args['output'], collage)
    plt.imshow(collage)
    plt.show()


if __name__ == '__main__':

    args = parse_opt()

    main(args)