"""
04-more-training
author: @maximellerbach
"""

import glob
import json
import os

import cv2
import numpy as np
import re
from tensorflow.keras.utils import Sequence
from sklearn import preprocessing 

def load_jsons(jsons_path):
    lines = []
    for json_path in jsons_path:
        with open(json_path, 'r') as file:
            lines += file.readlines()
    return lines

def load_image(img_path):
    img = cv2.imread(img_path)
    return img

def fill_table_with_data(directory_path):
    image_paths = []
    char_table = []
    for subdirectory in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory)
        if os.path.isdir(subdirectory_path):
            for filename in os.listdir(subdirectory_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
                    char_table.append(subdirectory)  # Collect subdirectory names
                    image_path = os.path.join(subdirectory_path, filename)
                    image_paths.append(image_path)
    return image_paths, char_table

class DataGenerator(Sequence):
    def __init__(
        self,
        data_directory,
        transform_funcs,
        batch_size=32,
    ):
        self.transform_funcs = transform_funcs
        self.batch_size = batch_size

        self.image_paths, self.characters_list = fill_table_with_data(data_directory)
        assert len(self.image_paths) > 0, "no images in directory were found"
        assert len(self.characters_list) > 0, "no characters in directory were found"
        # self.result = list(map(json.loads, load_jsons(self.json_paths)))
        # self.skip = sorted(json.loads(load_jsons(manifest)[-1])["deleted_indexes"], reverse=True)
        # for i in self.skip:
        #     del self.image_paths[i]
        #     del self.result[i]
        self.length = len(self.image_paths)
        self.len = self.length // self.batch_size + 1
        # # just check that every img / json paths does match

        # for (img_p, json_p) in zip(self.image_paths, self.json_paths):
        #     img_name = img_p.split(os.path.sep)[-1].split(".jpg")[0]
        #     json_name = json_p.split(os.path.sep)[-1].split(".catalog")[0]

        # assert img_name, json_name

    def __load_next(self):
        """Prepare a batch of data for training.

        X represents the input data, and Y the expected outputs (as in Y=f(X))

        Returns:
            tuple(list, list): X and Y.
        """

        X = []
        Y = []
        Z = []
        index = []

        l = np.random.randint(0, self.length, size=self.batch_size)
        label_encoder = preprocessing.LabelEncoder() 
        for i in l:
            img_path = self.image_paths[i]
            char = self.characters_list[i]
            image = load_image(img_path)

            for func in self.transform_funcs:
                image = func(image)

            image = cv2.resize(image, (240, 180))  # Adjust dimensions as needed
            # print(f"Image path: {img_path}, Image shape: {image.shape}, Image value: {char}")

            X.append(image)
            Y.append(label_encoder.fit_transform([char]))
            # Y.append(int(char))  # or Y.append(float(char_label))

        
        X = np.array(X) / 255.0  # Normalize pixel values to [0, 1]
        Y = np.array(Y)

        return X, Y

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.__load_next()


"""
Now some transform / data augmentation functions, 
they should only activate some of the time, for example 50% for the flip.
To do so, you can use np.random.random() and check if it is below a certain threshold.
I know this is not the most elegant/performant way, but it is easy to read and understand.

you can get creative with those !
Here are two of them, you can use many more !
"""


def flip(image: np.ndarray):
    """Simple image flip. Also flip the label."""

    rand = np.random.random()
    if rand < 0.5: # 50%
        image = cv2.flip(image, 0)

    return image


def noise(image: np.ndarray, mult=10):
    """Add some noise to the image."""
    
    rand = np.random.random()
    if rand < 0.1: # 10%
        noise = np.random.randint(-mult, mult, dtype=np.int8)
        image = image + noise # not perfect, here there could be some unsigned overflow 

    return image

def fast_clear(input_image):
    ''' input_image:  color or grayscale image
        brightness:  -255 (all black) to +255 (all white)

        returns image of same type as input_image but with
        brightness adjusted'''
    rand = np.random.random()
    if rand < 0.1: # 10%
        cv2.convertScaleAbs(input_image, input_image, 1, -50)
    return input_image

def fast_darkest(input_image):
    ''' input_image:  color or grayscale image
        brightness:  -255 (all black) to +255 (all white)

        returns image of same type as input_image but with
        brightness adjusted'''
    rand = np.random.random()
    if rand < 0.1: # 10%
        cv2.convertScaleAbs(input_image, input_image, 1, 50)
    return input_image

def saturation(image):
    rand = np.random.random()
    if rand < 0.1: # 10%
        image = image.astype(np.float32)  # Convert to float32
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 1] = hsv_image[..., 1] * 1.5
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image

def blur(image):
    rand = np.random.random()
    if rand < 0.1: # 10%
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

