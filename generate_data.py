import cv2
import numpy as np
import argparse

import os
from pathlib import Path
import random

import albumentations as A
from tqdm import tqdm


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--train_num', type=int, default=0)
    ap.add_argument('--dev_num', type=int, default=20)
    ap.add_argument('--page', type=str, default=os.path.join('data', 'page'))
    ap.add_argument('--foot', type=str, default=os.path.join('data', 'foot'))
    ap.add_argument('--background', type=str, default=os.path.join('data', 'background'))

    args = vars(ap.parse_args())

    return args


IMG_EXTS = ['.jpg', '.jpeg', '.png']


def make_dir(*args):
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


def mask_page(pg_img):
    # TODO cải tiến tiếp

    Z = pg_img.reshape(-1, 3)
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # for demo only
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res = res.reshape(pg_img.shape)

    mask = 255 - label.reshape(pg_img.shape[0], pg_img.shape[1]).astype('uint8') * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)

    return mask


def mask_foot(ft_img):
    # TODO cải tiến tiếp

    hsv = cv2.cvtColor(ft_img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = cv2.medianBlur(hsv[:, :, 0], 3)

    mask = np.where(np.logical_and(0 < hsv[:, :, 0], hsv[:, :, 0] < 20), 255, 0).astype('uint8')    # TODO xem lại đoạn này, bỏ cái 0 < đi được không?
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=5)

    return mask

def get_bg_transform(bg_paths, p=0.5):
    return A.Compose([
        A.RandomResizedCrop(height=512, width=512, scale=(0.01, 1.0), ratio=(0.2, 1.8), p=1),   # square 512 is critical
        A.RandomGridShuffle(p=p),
        A.Flip(p=p),
        A.Perspective(p=p),
        A.PiecewiseAffine(p=p),
        A.SafeRotate(p=p),
        A.Downscale(scale_min=0.5, scale_max=0.999, p=p),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=80, p=p),
        # A.FDA(bg_paths, p=p),
        # A.HistogramMatching(bg_paths, p=p),
        ])


def get_pg_transform(p=0.5):
    return A.Compose([
        A.LongestMaxSize(max_size=512, p=1),
        A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.SafeRotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.Perspective(fit_output=True, p=p),
        A.RandomResizedCrop(512, 512, scale=(0.35, 1.5), ratio=(1.0, 1.0), p=1),
        A.RandomShadow(p=p),
        A.Downscale(scale_min=0.5, scale_max=0.999, p=p),
    ])


def get_ft_transform(p=0.5):
    return A.Compose([
        A.LongestMaxSize(max_size=512, p=1),
        A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.RandomResizedCrop(512, 512, scale=(1.0, 1.0), ratio=(1.0, 1.0), p=1),
        A.Perspective(p=p),
        A.Downscale(scale_min=0.5, scale_max=0.999, p=p),
        A.HueSaturationValue(p=p),
    ])


def paste(src, des, mask):
    des = des.copy()
    des[mask == 255] = src[mask == 255]
    return des

def main(args):

    # read image paths
    bg_paths = [str(_) for _ in Path(args['background']).glob('*')
                if os.path.splitext(os.path.basename(str(_)))[1] in IMG_EXTS]
    pg_paths = [str(_) for _ in Path(args['page']).glob('*')
                if os.path.splitext(os.path.basename(str(_)))[1] in IMG_EXTS]
    ft_paths = [str(_) for _ in Path(args['foot']).glob('*')
                if os.path.splitext(os.path.basename(str(_)))[1] in IMG_EXTS]

    # read image array BGR
    bg_images = [cv2.imread(path) for path in bg_paths]
    pg_images = [cv2.imread(path) for path in pg_paths]
    ft_images = [cv2.imread(path) for path in ft_paths]

    # get mask for page and foot
    pg_masks = [mask_page(img) for img in pg_images]
    ft_masks = [mask_foot(img) for img in ft_images]

    # make dir for trainset and devset
    train_img_dir = make_dir('trainset', 'images')
    train_lbl_dir = make_dir('trainset', 'labels')
    dev_img_dir = make_dir('devset', 'images')
    dev_lbl_dir = make_dir('devset', 'labels')

    for mode, img_dir, lbl_dir, num in (('train', train_img_dir, train_lbl_dir, args['train_num']), ('dev', dev_img_dir, dev_lbl_dir, args['dev_num'])):
        start = len(os.listdir(img_dir))

        for i in tqdm(range(num), ascii=True, desc='Generating for ' + mode):

            # create new augmented background
            bg_transform = get_bg_transform(bg_paths, 0.5)
            bg_transformed = bg_transform(image=cv2.cvtColor(random.choice(bg_images), cv2.COLOR_BGR2RGB))
            syn_bg = cv2.cvtColor(bg_transformed['image'], cv2.COLOR_RGB2BGR)

            # create new augment page
            idx = random.randrange(len(pg_images))
            pg_img, pg_msk = pg_images[idx], pg_masks[idx]
            pg_transform = get_pg_transform(0.5)
            pg_transformed = pg_transform(image=cv2.cvtColor(pg_img, cv2.COLOR_BGR2RGB), mask=pg_msk)
            syn_pg = cv2.cvtColor(pg_transformed['image'], cv2.COLOR_RGB2BGR)
            syn_pg_msk = pg_transformed['mask']

            # create new augment foot
            idx = random.randrange(len(ft_images))
            ft_img, ft_msk = ft_images[idx], ft_masks[idx]
            ft_transform = get_ft_transform(0.5)
            ft_transformed = ft_transform(image=cv2.cvtColor(ft_img, cv2.COLOR_BGR2RGB), mask=ft_msk)
            syn_ft = cv2.cvtColor(ft_transformed['image'], cv2.COLOR_RGB2BGR)
            syn_ft_msk = ft_transformed['mask']

            syn_img = paste(syn_pg, syn_bg, syn_pg_msk)
            syn_img = paste(syn_ft, syn_img, syn_ft_msk)

            name = ('000000' + str(start + i))[-6:]
            cv2.imwrite(os.path.join(img_dir, name + '.jpg'), syn_img)
            cv2.imwrite(os.path.join(lbl_dir, name + '_pg.jpg'), syn_pg_msk)
            cv2.imwrite(os.path.join(lbl_dir, name + '_ft.jpg'), syn_ft_msk)

            # TODO sinh thêm ảnh không có chân/giấy


if __name__ == '__main__':

    args = parse_opt()

    main(args)

    import matplotlib.pyplot as plt

    # path = 'data/page/2.jpg'
    #
    # img = cv2.imread(path)
    # mask = mask_foot(img)
    # plt.imshow(mask)
    # plt.show()
