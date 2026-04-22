"""
This script evaluates the trained deepfake detection model on a validation/test dataset.
It generates a classification report and a confusion matrix.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from config import VALID_DIR, IMAGE_SIZE, BATCH_SIZE

def evaluate_model():
    if not os.path.exists(VALID_DIR):
        print(f"Error: Validation directory '{VALID_DIR}' not found.")
        return

    print("Loading model...")
    try:
        model = tf.keras.models.load_model('deepfake_detection_model.keras')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading validation dataset...")
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        VALID_DIR,
        labels='inferred',
        label_mode='binary',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Generating predictions...")
    y_true = []
    y_pred = []

    # Iterate through the dataset to get true labels and predictions
    for images, labels in val_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 1. Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred_binary, target_names=['FAKE', 'REAL'])
    print(report)

    # 2. Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred_binary)
    print(cm)

    # 3. Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['FAKE', 'REAL'], 
                yticklabels=['FAKE', 'REAL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Deepfake Detection')
    plt.savefig('evaluation_results.png')
    print("\nConfusion matrix saved as 'evaluation_results.png'")

if __name__ == "__main__":
    evaluate_model()
