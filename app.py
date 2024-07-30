import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input # type: ignore

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Deep Img/model_xxeption.keras')
    return model

model = load_model()

# Streamlit interface
st.title('Deepfake Image Detection')
st.write("Upload an image to check if it's real or fake.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    prediction = model.predict(img_array)
    pred_class = 'Fake' if prediction > 0.5 else 'Real'
    st.write(f'This image is {pred_class}.')

    # If you want to display additional information like confidence score
    st.write(f'Confidence Score: {prediction[0][0]:.2f}')
