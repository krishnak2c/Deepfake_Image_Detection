"""
This script evaluates the trained deepfake detection model on a validation/test dataset.
It generates a comprehensive report with advanced industrial metrics including:
- ROC-AUC & Precision-Recall AUC
- Confusion Matrix Heatmap
- Inference Latency (ms/image)
- Cohen's Kappa & Specificity/Sensitivity
"""

import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, auc, cohen_kappa_score
)
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

    print("Generating predictions & measuring latency...")
    y_true = []
    y_pred_probs = []
    
    start_time = time.time()
    # Iterate through the dataset to get true labels and predictions
    for images, labels in val_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)
    end_time = time.time()

    y_true = np.array(y_true).flatten()
    y_pred_probs = np.array(y_pred_probs).flatten()
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    total_images = len(y_true)
    avg_latency = (end_time - start_time) / total_images

    # --- Metric Calculations ---
    accuracy = np.mean(y_true == y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    kappa = cohen_kappa_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # --- Visualizations ---
    print("Generating plots...")
    plt.figure(figsize=(18, 5))

    # Plot 1: Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['FAKE', 'REAL'], 
                yticklabels=['FAKE', 'REAL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Plot 2: ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC area = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Plot 3: Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR area = {auc(recall, precision):.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    
    # --- Print Final Report ---
    print("\n" + "="*30)
    print("   DETAILED TECHNICAL STATS")
    print("="*30)
    print(f"Overall Accuracy:   {accuracy*100:.2f}%")
    print(f"ROC-AUC Score:      {roc_auc:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:        {specificity:.4f}")
    print(f"Cohen Kappa:        {kappa:.4f}")
    print(f"Inference Latency:  {avg_latency*1000:.2f} ms/image")
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred_binary, target_names=['FAKE', 'REAL']))
    print("\nVisualization saved as 'evaluation_results.png'")

if __name__ == "__main__":
    evaluate_model()
