import cv2
import numpy as np
import argparse

import os
from pathlib import Path
import random

ap = argparse.ArgumentParser()

ap.add_argument('--train_num', type=int, default=1) # TODO 1000
ap.add_argument('--val_num', type=int, default=1) # TODO 200
ap.add_argument('--page', type=str, default=os.path.join('data', 'page'))
ap.add_argument('--foot', type=str, default=os.path.join('data', 'foot'))
ap.add_argument('--background', type=str, default=os.path.join('data', 'background'))

args = vars(ap.parse_args())

IMG_EXTS = ['.jpg', '.jpeg', '.png']


def make_dir(*args):
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


def mask_page(pg_img):
    # TODO
    mask = np.zeros_like(pg_img, dtype='uint8')
    return mask


def mask_foot(ft_img):
    # TODO
    mask = np.zeros_like(ft_img, dtype='uint8')
    return mask


def synthesize(bg_img, pg_img, pg_mask, ft_img, ft_mask):
    # TODO
    # augment bg
    # crop pg, crop ft
    # resize pg, ft to adapt bg size
    # geometric augment pg (same way with mask), ft (same way with mask)
    # color/quality augment pg, ft
    # past pg, ft to bg

    syn_img = bg_img
    syn_pg_mask = pg_mask
    syn_ft_mask = ft_mask

    return syn_img, syn_pg_mask, syn_ft_mask


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
    dev_img_dir = make_dir('devset', 'labels')

    # TODO do for devset
    start = len(os.listdir(train_img_dir))
    for i in range(args['train_num']):
        bg_img = random.choice(bg_images)
        idx = random.randrange(len(pg_images))
        pg_img, pg_msk = pg_images[idx], pg_masks[idx]
        idx = random.randrange(len(ft_images))
        ft_img, ft_msk = ft_images[idx], ft_masks[idx]

        syn_img, syn_pg_msk, syn_ft_msk = synthesize(bg_img, pg_img, pg_msk, ft_img, ft_msk)
        name = ('000000' + str(start + i))[-6:]
        cv2.imwrite(os.path.join(train_img_dir, name + '.jpg'), syn_img)
        cv2.imwrite(os.path.join(train_lbl_dir, name + '_1.jpg'), syn_pg_msk)
        cv2.imwrite(os.path.join(train_lbl_dir, name + '_2.jpg'), syn_ft_msk)


if __name__ == '__main__':

    main(args)
