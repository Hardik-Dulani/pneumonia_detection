

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import cv2
from keras.models import load_model
import io
st.set_option('client.showErrorDetails', False)  # Hide error details in the app

# Function to convert JPEG image to bytes
def jpeg_to_bytes(image):
    """
    Converts a Pillow Image object to bytes.

    Args:
        image (PIL.Image.Image): The image to convert.

    Returns:
        bytes: Byte representation of the image.
    """
    byte_io = io.BytesIO()
    image.save(byte_io, format='JPEG')
    return byte_io.getvalue()

# Preprocess the image bytes
def preprocess(image_bytes, img_size=50):
    """
    Preprocess a single JPEG image for model inference.

    Args:
        image_bytes (bytes): JPEG image in binary format.
        img_size (int): Size to which the image should be resized.

    Returns:
        np.array: Preprocessed image ready for prediction.
    """
    # Decode the image from bytes
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    
    if img is None:
        raise ValueError("Invalid image data")

    # Resize and normalize
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0  # Normalize pixel values

    # Add batch and channel dimensions (shape: (1, img_size, img_size, 1))
    return np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0)

# Prediction function
def predict(model_path, image_bytes):
    model = load_model(model_path)
    
    # Preprocess the image
    preprocessed_image = preprocess(image_bytes)
    
    # Predict
    prediction = model.predict(preprocessed_image)
    return 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'

# Streamlit app
st.title("Pneumonia Detection")

uploaded_file = st.file_uploader("Upload a JPEG Image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Convert the image to bytes for further processing
    image_bytes = jpeg_to_bytes(image)

    # Preprocess and predict the image using the pickled functions
    try:
        prediction_result = predict('trained_model.h5', image_bytes)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    else:
        # Display the prediction results
        st.subheader("Prediction")
        st.write(f"Predicted Class: {prediction_result}")
