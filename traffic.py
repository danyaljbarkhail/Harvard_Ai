import cv2 as cv
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

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
    images = []
    labels = []

    os_data_dir = os.path.abspath(data_dir)

    # Traverse the directory tree
    for (dir_path, dir_names, files) in os.walk(os_data_dir):
        
        # Skip loading the current dir
        if dir_path == data_dir:
            continue
        
        # Ensure the directory name is a valid integer label
        label = os.path.basename(dir_path)
        if not label.isdigit() or int(label) >= NUM_CATEGORIES:
            continue

        # Process images in each folder
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
    model = tf.keras.models.Sequential(
        [
            # Convolutional layer. Learn 32 filters using a 3x3 kernel.
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.MaxPool2D((2, 2)),

            # Apply second convolution and pooling
            layers.Conv2D(32, (2, 2), activation="relu"),
            layers.MaxPool2D((2, 2)),

            # Flatten units for classification.
            layers.Flatten(),

            # Add hidden layers with dropout
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),

            # Output layer with units for all categories
            layers.Dense(NUM_CATEGORIES, activation="softmax")
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
