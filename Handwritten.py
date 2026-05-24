# ==========================================
# HANDWRITTEN DIGIT RECOGNITION
# ==========================================

# Import Libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# LOAD MNIST DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# NORMALIZE IMAGES
train_images = train_images / 255.0
test_images = test_images / 255.0

# RESHAPE DATA FOR CNN
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# DISPLAY SAMPLE IMAGE
plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0]}")
plt.show()

# BUILD CNN MODEL
model = models.Sequential()

# First Convolution Layer
model.add(layers.Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(28,28,1)
))

# Pooling Layer
model.add(layers.MaxPooling2D((2,2)))

# Second Convolution Layer
model.add(layers.Conv2D(
    64,
    (3,3),
    activation='relu'
))

# Second Pooling
model.add(layers.MaxPooling2D((2,2)))

# Flatten Layer
model.add(layers.Flatten())

# Dense Layers
model.add(layers.Dense(64, activation='relu'))

# Output Layer
model.add(layers.Dense(10, activation='softmax'))

# COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN MODEL
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

# EVALUATE MODEL
test_loss, test_acc = model.evaluate(
    test_images,
    test_labels
)

print("\nTest Accuracy:", test_acc)

# MAKE PREDICTION
prediction = model.predict(test_images)

predicted_digit = np.argmax(prediction[0])

print("\nPredicted Digit:", predicted_digit)
print("Actual Digit:", test_labels[0])

# SHOW TEST IMAGE
plt.imshow(test_images[0], cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.show()


# SAVE MODEL

model.save("handwritten_digit_model.h5")

print("\nModel Saved Successfully!")