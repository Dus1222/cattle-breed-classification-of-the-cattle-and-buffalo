import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from PIL import Image

st.set_page_config(page_title="Breed Classification", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("breed_resnet50_model.h5")

model = load_model()

class_names = [f"Breed{i}" for i in range(1, 42)]

st.title("🐮 Breed Classification App")
st.write("Upload an image of a cattle/buffalo to predict the breed using ResNet50 model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def predict_breed(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    predicted_class = np.argmax(preds)
    confidence = np.max(preds) * 100

    return class_names[predicted_class], confidence

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("🔄 Processing...")

    breed, conf = predict_breed(img)
    st.success(f"### 🐂 Predicted Breed: **{breed}**")
    st.info(f"### 🔥 Confidence: **{conf:.2f}%**")
