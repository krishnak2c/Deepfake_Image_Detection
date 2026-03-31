import streamlit as st

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from config import IMAGE_SIZE
import os


# --- Page Configuration ---
st.set_page_config(
    page_title="Deepfake Detection",
    layout="wide"
)

# --- Model Loading ---
# This is cached to prevent reloading the model on every interaction.
@st.cache_resource
def load_deepfake_model():
    """
    Loads the trained deepfake detection model.
    Using tf.keras.models.load_model is the recommended way as it loads
    the model architecture, weights, and optimizer state.
    """
    print("Loading model...")
    try:
        # Load the model saved in the '.keras' format from train.py, which is the best model.
        model = tf.keras.models.load_model('deepfake_detection_model.keras')
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None

model = load_deepfake_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    - Resizes to (224, 224) as used in the new training script.
    - Converts to a numpy array.
    - Adds a batch dimension.
    """
    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array)
    # Use tf.image.resize to ensure consistency with training preprocessing
    resized_tensor = tf.image.resize(image_tensor, IMAGE_SIZE)
    # Add a batch dimension
    return tf.expand_dims(resized_tensor, axis=0)

# --- UI Layout ---

# Header
st.markdown("<h1 style='text-align: center; color: grey;'>Deepfake Image Detection</h1>", unsafe_allow_html=True)
st.markdown("---")


# Main content area
if model is not None:
    uploaded_file = st.file_uploader("Upload an image to check if it's a deepfake.", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption='Uploaded Image', use_container_width=True)

            with col2:
                st.write("### Prediction")
                with st.spinner('Analyzing the image...'):
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    # model.predict returns a 2D array, e.g., [[0.99]]. 
                    # We need to access the inner value with [0][0].
                    prediction = model.predict(processed_image)[0][0]

                    # The model outputs a probability of being "Real".
                    # 'REAL' was class 1, 'FAKE' was class 0.
                    is_real_probability = prediction
                    if is_real_probability > 0.5:
                        result = "Real"
                        confidence = is_real_probability * 100
                        color = "green"
                        st.markdown(f"<h2 style='color:{color};'>This image appears to be REAL</h2>", unsafe_allow_html=True)
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        st.info("Our model is confident that this image is authentic. It does not show typical signs of manipulation associated with deepfakes.")

                    else:
                        result = "Fake"
                        confidence = (1 - is_real_probability) * 100
                        color = "red"
                        st.markdown(f"<h2 style='color:{color};'>This image appears to be a FAKE</h2>", unsafe_allow_html=True)
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        st.warning("Our model has detected characteristics commonly found in deepfakes. This could include subtle artifacts in facial features, lighting, or textures that suggest digital alteration.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    st.error("Model could not be loaded. The application cannot proceed.")


st.markdown("---")

# Informational Sections
st.markdown("### About Deepfakes")
st.write("""
Deepfakes are synthetic media in which a person in an existing image or video is replaced with someone else's likeness.
This technology uses powerful AI and deep learning techniques to create realistic fake content. While it has creative applications,
it also poses a significant threat in the form of misinformation, fake news, and malicious impersonation.
Our tool is designed to help identify such manipulated media to promote digital authenticity.
""")

st.markdown("### About the Dataset")
st.write("""
This model was trained on a comprehensive dataset of nearly 140,000 images to learn the differences between real and fake faces.
- **70,000 REAL images:** Sourced from the high-quality Flickr-Faces-HQ (FFHQ) dataset.
- **70,000 FAKE images:** Artificially generated using a StyleGAN model.

To prepare the data for training, all images are resized to 224x224 pixels to match the input size required by the underlying `EfficientNet` architecture. This large and balanced dataset is crucial for training a robust and accurate detector.
""")

st.markdown("### Model Training Performance")
st.write("The charts below show the model's performance during the training process on the validation dataset.")
# Check if the training history image exists before trying to display it.
if os.path.exists('training_history.png'):
    st.image('training_history.png')
else:
    st.info("Training history graph not found. Run the `train.py` script to generate the model and the training history graph.")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey;">
    <p>For educational and demonstration purposes.</p>
</div>
""", unsafe_allow_html=True)