import streamlit as st
from PIL import Image
import cv2


st.title('Foot size estimation')

file = st.file_uploader('Upload image')

if file:
    img = Image.open(file)
    cv2.imwrite('demo.jpg', img)
    img = cv2.imread('demo.jpg')
    st.image(Image.fromarray(img))