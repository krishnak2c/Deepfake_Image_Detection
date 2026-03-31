"""
This script trains a deepfake detection model using a two-stage process:
1. Feature Extraction: Trains a classifier on top of a frozen base model.
2. Fine-Tuning: Unfreezes the entire model and continues training with a lower learning rate.

Optimized for Google Colab with a T4 GPU.
- Uses TensorFlow and Keras with EfficientNetB0.
- Implements a tf.data pipeline for efficient data loading.
- Uses mixed precision to speed up training.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from config import (
    TRAIN_DIR,
    VALID_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS
)

# --- Configuration ---
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

def build_dataset(train_dir, valid_dir, image_size, batch_size):
    """Builds and augments training and validation datasets."""
    print("Building datasets...")
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        image_size=image_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir,
        labels='inferred',
        label_mode='binary',
        image_size=image_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        seed=42
    )

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    print("Datasets built successfully.")
    return train_dataset, validation_dataset

def build_model(image_size):
    """Builds the deepfake detection model with EfficientNetB0 base."""
    print("Building model...")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
    base_model.trainable = False  # Start with the base model frozen

    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
    model = Model(inputs, outputs)
    print("Model built successfully.")
    return model, base_model

def plot_history(history_initial, history_fine, initial_epochs):
    """Plots the combined training and validation history of both training stages."""
    acc = history_initial.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_initial.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history_initial.history['loss'] + history_fine.history['loss']
    val_loss = history_initial.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig('training_history.png')
    print("Combined training history plot saved as training_history.png")

def main():
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory '{TRAIN_DIR}' not found.")
        return

    strategy = tf.distribute.get_strategy()
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")

    train_ds, val_ds = build_dataset(TRAIN_DIR, VALID_DIR, IMAGE_SIZE, BATCH_SIZE)

    with strategy.scope():
        model, base_model = build_model(IMAGE_SIZE)
        model.compile(
            optimizer=Adam(learning_rate=0.001), # Higher LR for initial phase
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    print("--- Starting Feature Extraction Phase ---")
    model.summary()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    history_initial = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )

    print("\n--- Starting Fine-Tuning Phase ---")
    base_model.trainable = True # Unfreeze the base model
    # It's important to re-compile the model after making a layer non-trainable
    with strategy.scope():
        model.compile(
            optimizer=Adam(learning_rate=1e-5), # Use a very low learning rate for fine-tuning
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    model.summary()
    fine_tune_callbacks = [
        ModelCheckpoint('deepfake_detection_model.keras', save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]
    history_fine = model.fit(
        train_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history_initial.epoch[-1] + 1,
        validation_data=val_ds,
        callbacks=fine_tune_callbacks
    )
    
    print("Training finished.")
    # The best model is already saved by the ModelCheckpoint callback
    # as 'deepfake_detection_model.keras'.
    plot_history(history_initial, history_fine, INITIAL_EPOCHS)

if __name__ == '__main__':
    main()
