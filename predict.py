import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


import torch
from torchvision.transforms import ToTensor
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--weights', type=str, default='weights/ckpt.pth')
    ap.add_argument('--input', type=str, default=None)
    ap.add_argument('--output', type=str, default='output')

    args = vars(ap.parse_args())

    return args

#
# def iou(mask1, mask2):
#     union = np.zeros_like(mask1)
#     union[np.logical_or(mask1 == 255, mask2 == 255)] = 255
#     intersection = np.zeros_like(mask1)
#     intersection[np.logical_and(mask1 == 255, mask2 == 255)] = 255
#
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1); plt.imshow(union)
#     plt.subplot(1, 2, 2); plt.imshow(intersection)
#     plt.show()
#
#     retval, labels, stats, centroids = cv2.connectedComponentsWithStats(union, connectivity=4)
#     union_area = sum([stats[k, cv2.CC_STAT_AREA] for k in range(1, retval)])
#
#     retval, labels, stats, centroids = cv2.connectedComponentsWithStats(intersection, connectivity=4)
#     intersection_area = sum([stats[k, cv2.CC_STAT_AREA] for k in range(1, retval)])
#
#     return intersection_area/union_area


def arrange_corners(corners):
    """
    Arrange corner in the order: top-left, bottom-left, bottom-right, top-right
    :param corners: numpy array of shape (4, 1, 2)
    :return: numpy array of shape (4, 1, 2)
    """
    shape = corners.shape
    corners = np.squeeze(corners).tolist()
    corners = sorted(corners, key=lambda x: x[0])
    corners = sorted(corners[:2], key=lambda x: x[1]) + sorted(corners[2:], key=lambda x: x[1], reverse=True)
    return np.array(corners).reshape(shape)

def main(args):

    if not os.path.isdir(args['output']):
        os.makedirs(args['output'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = deeplabv3_mobilenet_v3_large(pretrained=True)
    net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    net.to(device)

    net.load_state_dict(torch.load(args['weights'], map_location=device))

    args['input'] = '/home/tran/Downloads/foots_all/images/23823.jpeg' #24071 23805 23808 23809 23810 23822 23823 23825 23829

    img = cv2.imread(args['input'])
    scale = 350/max(img.shape)
    W, H = int(img.shape[1] * scale), int(img.shape[0] * scale)
    img = cv2.resize(img, (W, H))
    x = ToTensor()(img[:, :, ::-1].copy()).unsqueeze(0)

    net.eval()
    with torch.no_grad():
        y = torch.sigmoid(net(x)['out'])

    mask = (y.cpu().squeeze().numpy() * 255).astype('uint8')

    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1); plt.imshow(mask)

    choices = []
    for thresh in range(254, 0, -1):
        _, threshed = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(threshed, connectivity=4)
        if 2 <= retval <= 4: # including background
            choices.append(thresh)
    thresh = int(np.mean(choices))
    _, mask = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)

    plt.subplot(2, 2, 2); plt.imshow(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(contours, axis=0))

    fail = True
    for alpha in np.arange(0.01, 0.5, 0.01):
        epsilon = alpha * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if approx.shape[0] == 4:
            fail = False
            break

    if fail:
        return

    demo = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), [approx], -1, (0, 255, 0))
    plt.subplot(2, 2, 3); plt.imshow(demo)

    approx = arrange_corners(approx)
    l = [np.sqrt(np.sum((approx[i] - approx[(i + 1) % 4]) ** 2)) for i in range(4)]
    if np.argmax(l) % 2 == 1:
        w, h = 297, 210
    else:
        w, h = 210, 297
    dst = np.array([
        [0, 0],
        [0, h],
        [w, h],
        [w, 0]
    ])
    M = cv2.getPerspectiveTransform(approx.astype('float32'), dst.astype('float32'))
    mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=15)

    mask = 255 - mask
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    k = sorted([(k, stats[k, cv2.CC_STAT_AREA]) for k in range(1, retval)], key=lambda x: x[1], reverse=True)[0][0]
    (x, y), (w, h), alpha = cv2.minAreaRect(np.roll(np.where(labels == k), 1, axis=0).transpose().reshape(-1, 1, 2))

    demo = cv2.drawContours(cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR), [np.int0(cv2.boxPoints(((x, y), (w, h), alpha)))], -1, (0, 255, 0))
    plt.subplot(2, 2, 4); plt.imshow(demo)

    plt.savefig('demo.png')

    plt.tight_layout()
    plt.show()
    print((297 - 2 * (297 - max(w, h)))/10)


if __name__ == '__main__':

    args = parse_opt()

    main(args)
