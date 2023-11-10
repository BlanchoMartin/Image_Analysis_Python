import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import DataGenerator, flip, noise, fast_clear, fast_darkest, saturation, blur

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'Train/')

# Iterate through subdirectories and collect paths of all images
def fill_table_with_data(directory_path, char_table, image_paths):
    for subdirectory in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory)
        if os.path.isdir(subdirectory_path):
            for filename in os.listdir(subdirectory_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
                    char_table.append(subdirectory)  # Collect subdirectory names
                    image_path = os.path.join(subdirectory_path, filename)
                    image_paths.append(image_path)

def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(180, 240, 3))
    x = tf.keras.layers.Cropping2D(((67, 0), (0, 0)))(inputs)
    x = tf.keras.layers.Conv2D(16, 3, 2, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, 2, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(48, 3, 2, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, 2, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    angle = tf.keras.layers.Dense(10, activation="linear", name="angle")(x)

    model = tf.keras.Model(inputs=inputs, outputs=angle)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model(model: tf.keras.Model, datagen: DataGenerator, epochs=10):
    """
    Provide the datagen object to the fit function as source of data.

    (Bonus) It is also a good idea to use a test_datagen to monitor the performance of the model on unseen data.
    In order to do that, you would have to create a new DataGenerator object with a different data directory.
    """
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(
        datagen,
        steps_per_epoch=len(datagen),
        epochs=epochs,
        callbacks=[reduce_lr,es_callback]
    )


def load_model(model_path):
    """"Simply load the model."""
    return tf.keras.models.load_model(model_path)


def predict(model, data_path):
    """
    Predict the steering angle for each image in the data_path.
    You can sort the images by name (date) to get the correct order then play the images as a video.

    hint: you can use cv2 to display the images
    You can also draw a visualisation of the steering angle on the image.
    """
    def sort_func(x):
        return float(x.split(os.path.sep)[-1].split("_")[0])

    for path in data_path:
        img = cv2.imread(path)
        img = cv2.resize(img, (240, 180))
        x = img / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model(x)[0][0]

        # draw horizontal line
        cv2.line(img, (80, 110), (int(80 + pred * 40), 110), (0, 0, 255), 4)

        # display image
        cv2.imshow("img", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    # model = load_model("trained_model.h5") #-- visuel
    model = build_model()
    model.summary()

    # if you have not implemented any transform funcs yet, just put an empty list []
    datagen = DataGenerator(data_path, [], batch_size=128)

    # if the traning takes too much time, you can try to reduce the batch_size and the number of epochs
    # tf.profiler.experimental.start('logs')
    train_model(model, datagen, epochs=5)
    # tf.profiler.experimental.stop()

    model.save("trained_model.h5")
    # predict(model, "/home/guigui/Epitech-Robotcar/practices/04-more-training/data_test/data/images") #-- visuel