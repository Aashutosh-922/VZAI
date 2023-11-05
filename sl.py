import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to grayscale, resize, and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(-1, 28, 28, 1)

# Load the model
model = tf.keras.models.load_model('handwritten_text_classifier.h5')

# Set up the title of the app
st.title('Handwritten Text Classification')

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    # Preprocess the image and predict
    image = preprocess_image(np.array(image))
    prediction = model.predict(image)
    st.write(f'Predicted class: {np.argmax(prediction)}')
