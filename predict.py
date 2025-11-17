import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
# The model is loaded with the custom objects needed for mixed precision
model = load_model('deepfake_detection_model.h5')

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # --- BUG FIX ---
    # The model was trained on 128x128 images. Changed from 96x96.
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # --- OPTIMIZATION ---
    # Use the same preprocessing as in training for better accuracy.
    image = preprocess_input(image)
    return image

# Predict if the image is fake or real
def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    class_label = np.argmax(prediction, axis=1)[0]
    return "Fake" if class_label == 0 else "Real"

# --- IMPORTANT ---
# Change this path to your image location in Google Colab
image_path = "real_and_fake_face_detection/real_and_fake_face/training_real/real_00001.jpg"
if os.path.exists(image_path):
    result = predict_image(image_path)
    print(f"The image is {result}")
else:
    print(f"Error: Image path not found at '{image_path}'")