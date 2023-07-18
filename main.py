import streamlit as st
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


def get_compressed(image,y):
  input_img = plt.imread(image)
  img_data = (input_img / 255.0).reshape(-1, 3)
  kmeans = MiniBatchKMeans(32).fit(img_data)
  k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]
  k_img = np.reshape(k_colors, (input_img.shape))
  return k_img

title = st.title('Image compressor')

st.subheader('Upload your image file')

uploaded_file = st.file_uploader(label = "choose your image",type = ['jpg'])
y = st.slider('select compress level(with '1' with highest compress)',1,100,32)
if uploaded_file is not None:
    byte_data = uploaded_file.getvalue()
    st.image(Image.open(uploaded_file),width=750)
    file_size = len(byte_data)
    st.write("Filesize: ", round(file_size/1000),"KB" )

Compress = st.button('Compress image')

if Compress:
    compressed_image =  get_compressed(uploaded_file,y)
    img = Image.fromarray((compressed_image * 255).astype(np.uint8))
    st.image(img,width=750)

    img.save('compressed_image.jpg')
    im_size = os.stat('compressed_image.jpg').st_size
    st.write("Compressed filesize: ", round(im_size/1024) , "KB")

    with open("compressed_image.jpg", "rb") as file:
        btn = st.download_button(
                label="Download image",
                data=file,
                file_name="compressed_image.jpg",
                mime="image/png"
            )
