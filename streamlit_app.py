import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import ToTensor

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

st.title('Foot size estimation')

st.write("""[github](https://github.com/LapTQ/foot_size_estimation)""")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = deeplabv3_mobilenet_v3_large(pretrained=True)
net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
net.to(device)
net.load_state_dict(torch.load('weights/ckpt.pth', map_location=device))

file = st.file_uploader('Upload image')

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

def main():
    if file:
        img = Image.open(file)
        st.image(img)

        img = np.array(img.convert('RGB'))
        scale = 350 / max(img.shape)
        W, H = int(img.shape[1] * scale), int(img.shape[0] * scale)
        img = cv2.resize(img, (W, H))

        x = ToTensor()(img.copy()).unsqueeze(0)

        net.eval()
        with torch.no_grad():
            y = torch.sigmoid(net(x)['out'])

        mask = (y.cpu().squeeze().numpy() * 255).astype('uint8')

        plt.figure(figsize=(20, 20))
        plt.subplot(2, 2, 1);
        plt.imshow(mask)

        choices = []
        for thresh in range(254, 0, -1):
            _, threshed = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(threshed, connectivity=4)
            if 2 <= retval <= 4:  # including background
                choices.append(thresh)
        thresh = int(np.mean(choices))
        _, mask = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)

        plt.subplot(2, 2, 2);
        plt.imshow(mask)

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
            st.write('Cannot estimate with this image. Please capture again.')
            return

        demo = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), [approx], -1, (0, 255, 0))
        plt.subplot(2, 2, 3);
        plt.imshow(demo)

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

        demo = cv2.drawContours(cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR),
                                [np.int0(cv2.boxPoints(((x, y), (w, h), alpha)))], -1, (0, 255, 0))
        plt.subplot(2, 2, 4);
        plt.imshow(demo)

        plt.tight_layout()
        plt.savefig('demo.png')

        st.image(Image.open('demo.png'))
        st.write(f'Estimated size: {(297 - 2 * (297 - max(w, h))) / 10}')

try:
    main()
except:
    st.write('Cannot estimate with this image. Please capture again.')