import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define activation functions to use
activation_functions = ['sigmoid', 'relu', 'leaky_relu']

# Loop through different activation functions
for activation_func in activation_functions:
    print(f"\nTraining model with {activation_func.upper()} activation function:")

    # Build a simple convolutional neural network (CNN) model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation=activation_func, input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activation_func))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activation_func))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=activation_func))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
    print(f"Test accuracy with {activation_func.upper()} activation function: {test_acc}")

    # Save the model
    model.save(f"mnist_model_{activation_func}.h5")

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.title(f'Training History - {activation_func.upper()} Activation Function')
    plt.legend()
    plt.show()

    # Display predictions for the first 10 test images
    # for i in range(10):
    #     img = test_images[i]
    #     true_label = test_labels[i]

    #     # Make a prediction
    #     prediction = model.predict(np.expand_dims(img, axis=0).reshape(-1, 28, 28, 1))[0]
    #     predicted_label = np.argmax(prediction)

    #     # Display the image
    #     img_display = cv2.resize(img, (200, 200))
    #     cv2.imshow(f"{activation_func.upper()} Activation - True label: {true_label}, Predicted label: {predicted_label}", img_display)

cv2.waitKey(0)
cv2.destroyAllWindows()
