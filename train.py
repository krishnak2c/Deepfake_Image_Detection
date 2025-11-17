import numpy as np
import pandas as pd
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm
import os

# Define paths
real = input("Enter the path to the real dataset (e.g., real_and_fake_face_detection/real_and_fake_face/training_real/): ")
fake = input("Enter the path to the fake dataset (e.g., real_and_fake_face_detection/real_and_fake_face/training_fake/): ")

# Load image paths
real_path = os.listdir(real)
fake_path = os.listdir(fake)

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
    plt.imshow(load_img(real + real_path[i]), cmap='gray')
    plt.suptitle("Real faces", fontsize=20)
    plt.axis('off')
plt.show()

fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(fake + fake_path[i]), cmap='gray')
    plt.suptitle("Fake faces", fontsize=20)
    plt.title(fake_path[i][:4])
    plt.axis('off')
plt.show()

# TPU Strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)



AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 32 * strategy.num_replicas_in_sync



# Data pipeline

def get_dataset(real_path, fake_path):



    real_images = [os.path.join(real, img) for img in os.listdir(real)]



    fake_images = [os.path.join(fake, img) for img in os.listdir(fake)]



    



    all_images = real_images + fake_images



    labels = [1] * len(real_images) + [0] * len(fake_images)



    



    return train_test_split(all_images, labels, test_size=0.2, random_state=42)







def decode_image(image_path, label):



    image = tf.io.read_file(image_path)



    image = tf.image.decode_jpeg(image, channels=3)



    image = tf.image.resize(image, [128, 128])



    image = tf.image.convert_image_dtype(image, tf.float32)



    return image, label







def augment(image, label):



    image = tf.image.random_flip_left_right(image)



    image = tf.image.random_brightness(image, max_delta=0.1)



    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)



    return image, label







def create_dataset(image_paths, labels, is_training=True):



    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))



    dataset = dataset.map(decode_image, num_parallel_calls=AUTO)



    if is_training:



        dataset = dataset.map(augment, num_parallel_calls=AUTO)



    dataset = dataset.shuffle(buffer_size=len(image_paths))



    dataset = dataset.batch(BATCH_SIZE)



    dataset = dataset.prefetch(buffer_size=AUTO)



    return dataset







X_train, X_val, y_train, y_val = get_dataset(real, fake)







train_dataset = create_dataset(X_train, y_train)



val_dataset = create_dataset(X_val, y_val, is_training=False)







# MobileNetV2 model



with strategy.scope():



    mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(128, 128, 3))



    tf.keras.backend.clear_session()







    model = Sequential([mnet,



                        GlobalAveragePooling2D(),



                        Dense(512, activation="relu"),



                        BatchNormalization(),



                        Dropout(0.5),



                        Dense(256, activation="relu"),



                        BatchNormalization(),



                        Dropout(0.3),



                        Dense(128, activation="relu"),



                        Dropout(0.1),



                        Dense(2, activation="softmax")])







    for layer in mnet.layers[-20:]:



        layer.trainable = True







    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])







model.summary()







# Callbacks



early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)



model_checkpoint = ModelCheckpoint('deepfake_detection_model.h5', save_best_only=True)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)







hist = model.fit(train_dataset,



                 epochs=50,



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
plt.savefig('Figure_1.png')

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'], loc=4)
plt.style.use(['classic'])
plt.savefig('Figure_2.png')
