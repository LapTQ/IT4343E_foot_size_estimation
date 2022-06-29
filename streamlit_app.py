import streamlit as st
from PIL import Image
import cv2
import numpy as np

import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

st.title('Foot size estimation')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = deeplabv3_mobilenet_v3_large(pretrained=True)
net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
net.to(device)
net.load_state_dict(torch.load('weights/ckpt.pth', map_location=device))

file = st.file_uploader('Upload image')

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
        y = F.sigmoid(net(x)['out'])

    mask = (y.cpu().squeeze().numpy() * 255).astype('uint8')

    st.image(Image.fromarray(mask))