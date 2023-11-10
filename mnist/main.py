import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build a simple convolutional neural network (CNN) model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save("mnist_model.h5")

for i in range(10):  # Display predictions for the first 10 test images
    img = test_images[i]
    true_label = test_labels[i]

    # Make a prediction
    prediction = model.predict(np.expand_dims(img, axis=0).reshape(-1, 28, 28, 1))[0]
    predicted_label = np.argmax(prediction)

    # Display the image
    img_display = cv2.resize(img, (200, 200))
    cv2.imshow(f"True label: {true_label}, Predicted label: {predicted_label}", img_display)

cv2.waitKey(0)
cv2.destroyAllWindows()