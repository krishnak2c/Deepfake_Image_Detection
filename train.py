import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import cv2
from tqdm import tqdm
import os

# --- Optimized for T4 GPU on Google Colab ---
# 1. Enabled mixed precision for Tensor Core performance.
# 2. Removed TPU strategy, using default GPU strategy.
# 3. Hardcoded paths instead of input() for non-interactive execution.
# 4. Removed plt.show() to prevent blocking in non-interactive environments.

# Enable mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- IMPORTANT ---
# Change these paths to your dataset location in Google Colab
real = 'dataset/train/REAL/'
fake = 'dataset/train/FAKE/'

# Load image paths for visualization
real_path_viz = os.listdir(real)
fake_path_viz = os.listdir(fake)

# Derive dataset_path from the real path
dataset_path = os.path.dirname(os.path.dirname(real))
if not os.path.exists(dataset_path):
    print(f"Error: Dataset path '{dataset_path}' not found.")
    exit()

# Visualizing real and fake faces
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[..., ::-1]

fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(os.path.join(real, real_path_viz[i])), cmap='gray')
    plt.suptitle("Real faces", fontsize=20)
    plt.axis('off')
plt.savefig('real_faces_preview.png')

fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(os.path.join(fake, fake_path_viz[i])), cmap='gray')
    plt.suptitle("Fake faces", fontsize=20)
    plt.title(fake_path_viz[i][:4])
    plt.axis('off')
plt.savefig('fake_faces_preview.png')

# GPU Strategy
strategy = tf.distribute.get_strategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
SHUFFLE_BUFFER_SIZE = 1024 # Using a fixed buffer size to limit memory usage
VALIDATION_SPLIT = 0.2

# Data pipeline
def decode_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    # No conversion to float32 here, preprocess_input will handle it
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def configure_dataset(ds, is_training=True):
    ds = ds.map(decode_image, num_parallel_calls=AUTO)
    if is_training:
        ds = ds.map(augment, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTO)
    return ds

# Create datasets from file paths
real_files = tf.data.Dataset.list_files(os.path.join(real, '*.jpg'), shuffle=True)
fake_files = tf.data.Dataset.list_files(os.path.join(fake, '*.jpg'), shuffle=True)

# Get the number of files
num_real = tf.data.experimental.cardinality(real_files)
num_fake = tf.data.experimental.cardinality(fake_files)

real_labels = tf.data.Dataset.from_tensor_slices(tf.ones(num_real, dtype=tf.int32))
fake_labels = tf.data.Dataset.from_tensor_slices(tf.zeros(num_fake, dtype=tf.int32))

real_ds = tf.data.Dataset.zip((real_files, real_labels))
fake_ds = tf.data.Dataset.zip((fake_files, fake_labels))

dataset = real_ds.concatenate(fake_ds)
dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

dataset_size = num_real + num_fake
train_size = tf.cast(tf.cast(dataset_size, tf.float32) * (1 - VALIDATION_SPLIT), tf.int64)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

train_dataset = configure_dataset(train_dataset)
val_dataset = configure_dataset(val_dataset, is_training=False)

# MobileNetV2 model
with strategy.scope():
    mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    
    # Pre-process input for mobilenet
    inputs = tf.keras.layers.Input([128,128,3])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = mnet(x)

    model = Sequential([
                        GlobalAveragePooling2D(),
                        Dense(512, activation="relu"),
                        BatchNormalization(),
                        Dropout(0.5),
                        Dense(256, activation="relu"),
                        BatchNormalization(),
                        Dropout(0.3),
                        Dense(128, activation="relu"),
                        Dropout(0.1),
                        Dense(2, activation="softmax", dtype='float32') # Output layer should be float32
                        ])
    
    outputs = model(x)
    model = tf.keras.Model(inputs, outputs)

    for layer in mnet.layers[-20:]:
        layer.trainable = True

    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('deepfake_detection_model.h5', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

hist = model.fit(train_dataset,
                 epochs=8,
                 callbacks=[early_stopping, model_checkpoint, reduce_lr],
                 validation_data=val_dataset)

# Save model
model.save('deepfake_detection_model.h5')

# Visualizing accuracy and loss
epochs = len(hist.history['loss'])
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train', 'Validation'])
plt.style.use(['classic'])
plt.savefig('loss_vs_epochs.png')

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'], loc=4)
plt.style.use(['classic'])
plt.savefig('accuracy_vs_epochs.png')