import tensorflow as tf, keras
from keras._tf_keras.keras import layers, models
from keras._tf_keras.keras.applications import ResNet50

from keras._tf_keras.keras.optimizers import Adam

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.datasets import cifar10

# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(x_train)

# Define the CNN model
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

# Train the CNN model
#history_cnn = model_cnn.fit(x_train, y_train_onehot, epochs=20, batch_size=64,validation_data=(x_test, y_test_onehot))

history_cnn = model_cnn.fit(datagen.flow(x_train, y_train_onehot, batch_size=64),
                            epochs=20,
                            validation_data=(x_test, y_test_onehot))

# Predict on the test set
y_pred_cnn = model_cnn.predict(x_test)
y_pred_cnn = np.argmax(y_pred_cnn, axis=1)  # Convert probabilities to class labels

# Calculate metrics
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
precision_cnn = precision_score(y_test, y_pred_cnn, average='macro')
recall_cnn = recall_score(y_test, y_pred_cnn, average='macro')
f1_cnn = f1_score(y_test, y_pred_cnn, average='macro')

print("CNN Metrics:")
print(f"Accuracy: {accuracy_cnn:.4f}")
print(f"Precision: {precision_cnn:.4f}")
print(f"Recall: {recall_cnn:.4f}")
print(f"F1-Score: {f1_cnn:.4f}")

# Load ResNet50 with pre-trained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers
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

# Train the ResNet model
history_resnet = model_resnet.fit(x_train, y_train_onehot, epochs=10, batch_size=64,
                                  validation_data=(x_test, y_test_onehot))

# Predict on the test set
y_pred_resnet = model_resnet.predict(x_test)
y_pred_resnet = np.argmax(y_pred_resnet, axis=1)  # Convert probabilities to class labels

# Calculate metrics
accuracy_resnet = accuracy_score(y_test, y_pred_resnet)
precision_resnet = precision_score(y_test, y_pred_resnet, average='macro')
recall_resnet = recall_score(y_test, y_pred_resnet, average='macro')
f1_resnet = f1_score(y_test, y_pred_resnet, average='macro')

print("ResNet Metrics:")
print(f"Accuracy: {accuracy_resnet:.4f}")
print(f"Precision: {precision_resnet:.4f}")
print(f"Recall: {recall_resnet:.4f}")
print(f"F1-Score: {f1_resnet:.4f}")

test_loss_cnn, test_acc_cnn = model_cnn.evaluate(x_test, y_test_onehot)
print("CNN Test Accuracy:", test_acc_cnn)

test_loss_resnet, test_acc_resnet = model_resnet.evaluate(x_test, y_train_onehot)
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

