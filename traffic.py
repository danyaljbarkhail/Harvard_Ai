import cv2 as cv
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool2D

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    os_data_dir = os.path.join(os.getcwd(), data_dir)

    # Traverse the directory tree
    for (dir_path, dir_names, files) in os.walk(os_data_dir):
        
        # Skip loading the current dir
        if dir_path == data_dir:
            continue
        
        label = os.path.split(dir_path)[1]
        # For each folder, process and append their images
        for file_name in files:
            try:
                img = cv.imread(os.path.join(dir_path, file_name), 1)
                res = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)
            
                images.append(res)
                labels.append(int(label))

            except Exception as e:
                print(str(e))
    
    print(f"Processed {len(images)} images of {len(set(labels))} different types.")

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential(
        [
            # Convolutional layer. Learn 32 filters using a 3x3 kernel.
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

            # Max-pooling layer, using 2x2 pool size
            layers.MaxPool2D((2, 2)),

            # First convolution and pooling learns low-level features. Possibly curves and edges?

            # Apply second convolution and pooling
            layers.Conv2D(32, (2, 2), activation="relu"),
            layers.MaxPool2D((2, 2)),

            # Second iteration learns high-level features. Possibly road signs.

            # Flatten units for classification.
            layers.Flatten(),

            # Add a hidden layer with dropout
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),

            # Randomly dropout half of the hidden layer nodes to prevent overfitting
            layers.Dropout(0.5),

            # Output layer with units for all categories
            layers.Dense(NUM_CATEGORIES, activation="softmax")
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
