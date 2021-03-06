# Import libraries

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from PIL import Image
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adam

# Data exploration visualization

'''''
import matplotlib.pyplot as plt
plt.rcdefaults()
fig, ax1 = plt.subplots(1,1)
ax1.barh(np.arange(len(y_train)), y_train, align='center',
        color='green')
#ax1.set_yticks(np.arange(len(y_train)))
ax1.set_yticklabels(y_train)
ax1.set_ylabel('y_train')
#ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlabel('steerings')
ax1.set_title('Training set')
plt.show()
'''''

# Function for preprocessing images on the fly while they are put through generator
# TODO: Any changes made in this function have to be adopted to drive.py

def preprocessImage(image):

    # Cropping
    image = image[60:135, :, :]  #
    # Converting color due to cv2 import
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resizing to final size
    image = cv2.resize(image, (width_size_final, height_size_final))

    return image


def manipulateBrightness(image):

    # Converting image to HSV: adjusting brightness is accomplished by manipulating V channel
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Random change of brightness, from a starting point though, to avoid getting too dark
    delta_brightness = 0.2 + np.random.uniform()
    # Only manipulating V channel
    image[:, :, 2] = image[:, :, 2] * delta_brightness
    # Converting back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return image


# Generator for training data

def createGenerator_train(data_frame, batch_size):

    X_batch = np.zeros((batch_size, height_size_final, width_size_final, 3))
    y_batch = np.zeros(batch_size)
    count_images = 0  # used for creating image output
    flip = 0
    while 1:
        for b in range(batch_size):
            # start process of randomly picking and processing image till output steering value is above critical value
            # pick random line (1 line = 3 images + 1 steering angle)
            line_index = np.random.randint(len(data_frame))
            # pick random column containing images of different points of view: 0 = CENTER | 1 = LEFT | 2 = RIGHT
            column_index = np.random.randint(3)
            # load chosen image from image data directory
            image_dir = "data-udacity/" + data_frame.iloc[line_index, column_index].strip()
            image = cv2.imread(image_dir)
            # load corresponding steering value
            steering_offset = 0  # no steering offset for CENTER view (column_index==0)
            if column_index == 1:  # image of LEFT view was picked
                steering_offset = +0.2
            if column_index == 2:  # image of RIGHT view was picked
                steering_offset = -0.2
            steering_angle = data_frame.iloc[line_index, 3] + steering_offset
            # change brightness of image randomly
            image = manipulateBrightness(image)
            # flip every second image and steering angle to reduce left/right bias
            if flip < 1:  # change to adjust bias
                image = cv2.flip(image, 1)
                steering_angle *= -1.0
                flip += 1
            else:
                flip = 0
            # preprocess image data and assign steering angle
            X_batch[b] = preprocessImage(image)
            y_batch[b] = steering_angle
            '''''
            # save a couple of pictures to check preprocess output while testing
            if count_images > 999:
                file_dir = "Input-images/steering: " + str(steering_angle).strip() + ".jpg"
                mpimg.imsave(file_dir, X_batch[b])
                count_images = 0
            count_images += 1
            '''''

        yield X_batch, y_batch



# Generator for validation data

def createGenerator_valid(data_frame, batch_size):

    # Use a different generator for the validation set in order to keep the validation data as original as possible
    X_batch = np.zeros((batch_size, height_size_final, width_size_final, 3))
    y_batch = np.zeros(batch_size)
    while 1:
        for b in range(batch_size):
            # start process of randomly picking and image
            # pick random line (1 line = 3 images + 1 steering angle)
            line_index = np.random.randint(len(data_frame))
            # pick CENTER image (as this is what the simulator is using in autonomous mode
            column_index = 0
            # load chosen image from image data directory
            image_dir = "data-udacity/" + data_frame.iloc[line_index, column_index].strip()
            image = cv2.imread(image_dir)
            # load corresponding steering value
            steering_angle = data_frame.iloc[line_index, 3]
            # Resizing to fit model, other than that no preprocessing applied
            image = cv2.resize(image, (width_size_final, height_size_final))
            X_batch[b] = image
            y_batch[b] = steering_angle

        yield X_batch, y_batch


# Architecture inspired by NVIDIA's paper "End to End Learning for Self-Driving Cars"

def createModel():

    model = Sequential()
    # Normalizing
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(height_size_final, width_size_final, 3)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))
#   model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))
#   model.add(MaxPooling2D())

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('elu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('elu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # Configure learning process
    model.compile(optimizer="adam", loss="mse")

    return model

# MAIN SECTION

# Final image size for input. Has to be same in drive.py!
height_size_final = 64
width_size_final = 64

# Read in CSV
data_frame = pd.read_csv('data-udacity/driving_log.csv', usecols=[0, 1, 2, 3])

# Validation split
valid_share = 0.2
split = int(data_frame.shape[0] * (1 - valid_share))
data_frame_train = data_frame.loc[0:split-1]
data_frame_valid = data_frame.loc[split:]

# Create network architecture
model = createModel()

# Train model with generator
BATCH_SIZE = 32
SAMPLES_PER_EPOCH = (24108//BATCH_SIZE)*BATCH_SIZE

generator_train = createGenerator_train(data_frame_train, BATCH_SIZE)
generator_valid = createGenerator_valid(data_frame_valid, BATCH_SIZE)

history = model.fit_generator(generator_train, validation_data=generator_valid, nb_val_samples=4821,
                              samples_per_epoch=SAMPLES_PER_EPOCH, nb_epoch=8)

# Save training results
model.save('model.h5')
print("Model saved.")
