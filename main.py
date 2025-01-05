import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('C:/Users/admin/OneDrive/Desktop/AI_ML_ipynb/minor_fake_image/model_densenet.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction
def make_prediction(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]

# Streamlit app UI
st.title("Image Authenticity Detector: Real or Fake")

st.write("You can either upload an image or provide an image URL to classify it.")

# Image upload option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# URL input option
image_url = st.text_input("Or enter the URL of an image:")

image = None

if uploaded_file is not None:
    # If the user uploads an image, load and display it
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

elif image_url:
    try:
        # Fetch and display the image from the URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Image from URL', use_column_width=True)
    except Exception as e:
        st.error(f"Unable to load image from the provided URL. Error: {e}")

# If an image is available (either uploaded or from URL), process it
if image is not None:
    st.write("Processing...")

    # Make prediction
    prediction = make_prediction(image)

    # Show result
    if prediction > 0.5:
        st.write("The image is **REAL** with {:.2f}% confidence.".format(prediction * 100))
    else:
        st.write("The image is **FAKE** with {:.2f}% confidence.".format((1 - prediction) * 100))

# Run the app using the command `streamlit run your_script_name.py`
