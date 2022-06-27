import streamlit as st
from PIL import Image
import cv2
import numpy as np

import torch
from torchvision.transforms import ToTensor

from models.unet import UNet

st.title('Foot size estimation')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(3, 2).to(device)
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

    y = np.where(net(x).squeeze() > 0.5, 1, 0).astype('uint8')
    pg_mask, ft_mask = y[0], y[1]

    collage = np.concatenate([pg_mask, ft_mask], axis=1)
    st.image(Image.fromarray(collage))