import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load the saved models in a loop
activation_functions = ['sigmoid', 'relu', 'leaky_relu']

for activation_func in activation_functions:
    # Load the saved model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"mnist_model_{activation_func}.h5")
    model = keras.models.load_model(model_path)

    # Load the MNIST test dataset
    (_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
    test_images = test_images / 255.0  # Normalize pixel values to be between 0 and 1

    # Reshape the test images to match the model input shape
    test_images = test_images.reshape(-1, 28, 28, 1)

    # Perform predictions on the test set
    predictions = model.predict(test_images)

    # Plot the first 10 test images along with their predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle(f'Model with {activation_func.upper()} Activation Function', fontsize=16)
    
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i].reshape(28, 28), cmap="gray")
        plt.title(f"True: {test_labels[i]}\nPred: {np.argmax(predictions[i])}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
