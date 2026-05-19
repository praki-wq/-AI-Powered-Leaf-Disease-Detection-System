import streamlit as st
from predict import predict_disease
from PIL import Image
import os

st.title("🌿 Leaf Disease Detection System")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    temp_path = "temp.jpg"

    image.save(temp_path)

    disease, confidence = predict_disease(temp_path)

    st.success(f"Disease: {disease}")

    st.info(f"Confidence: {confidence}%")

    os.remove(temp_path)