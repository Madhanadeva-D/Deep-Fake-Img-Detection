import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model_path = "D:/NTT/DeepFake Detection/model.keras"
model = load_model(model_path)

target_size = (256, 256)

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

st.title('Deepfake Image Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    img_path = os.path.join("temp", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    img, img_array = load_and_preprocess_image(img_path, target_size)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("")

    prediction = model.predict(img_array)

    real_confidence = prediction[0][0] * 100
    fake_confidence = (1 - prediction[0][0]) * 100

    if real_confidence > fake_confidence:
        st.write(f"### The model predicts this image is **REAL** with a confidence of {real_confidence:.2f}%.")
    else:
        st.write(f"### The model predicts this image is **FAKE** with a confidence of {fake_confidence:.2f}%.")

    st.write(f"Confidence Scores: **Real**: {real_confidence:.2f}% | **Fake**: {fake_confidence:.2f}%")


# python -m streamlit run app.py