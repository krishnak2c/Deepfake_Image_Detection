import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image):
    """
    Preprocesses an image for the model.
    """
    # --- BUG FIX ---
    # The model was trained on 128x128 images. Changed from 96x96.
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # --- OPTIMIZATION ---
    # Use the same preprocessing as in training for better accuracy.
    image = preprocess_input(image)
    return image

def predict_image(model, image):
    """
    Predicts if an image is fake or real.
    """
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_label = np.argmax(prediction, axis=1)[0]
    return "Fake" if class_label == 0 else "Real"