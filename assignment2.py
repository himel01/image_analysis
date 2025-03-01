import tensorflow as tf, keras
from keras._tf_keras.keras import layers, models
from keras._tf_keras.keras.applications import ResNet50

from keras._tf_keras.keras.optimizers import Adam

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Check dataset shape
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(x_train)

model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model_cnn.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Print model summary
model_cnn.summary()

history_cnn = model_cnn.fit(datagen.flow(x_train, y_train, batch_size=64),
                            epochs=20,
                            validation_data=(x_test, y_test))



base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
model_resnet = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model_resnet.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Print model summary
model_resnet.summary()

history_resnet = model_resnet.fit(datagen.flow(x_train, y_train, batch_size=64),
                                  epochs=10,
                                  validation_data=(x_test, y_test))

test_loss_cnn, test_acc_cnn = model_cnn.evaluate(x_test, y_test)
print("CNN Test Accuracy:", test_acc_cnn)

test_loss_resnet, test_acc_resnet = model_resnet.evaluate(x_test, y_test)
print("ResNet Test Accuracy:", test_acc_resnet)

# Plot CNN training history
plt.plot(history_cnn.history['accuracy'], label='CNN Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Validation Accuracy')
plt.plot(history_resnet.history['accuracy'], label='ResNet Training Accuracy')
plt.plot(history_resnet.history['val_accuracy'], label='ResNet Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()