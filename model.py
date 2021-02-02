import csv
import cv2
import numpy as np

lines = []
with open('training/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'training/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    # Data augmentation
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Speed up training by cropping images
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
# Network architecture from NVIDIA
# Layer 0: Normalization.
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# Layer 1: Convolutional. Filters = 24, Filter Size = 5x5, Strides = 2x2.
model.add(Convolution2D(24, 5, strides=(2, 2), activation="elu"))
# Layer 2: Convolutional. Filters = 36, Filter Size = 5x5, Strides = 2x2.
model.add(Convolution2D(36, 5, strides=(2, 2), activation="elu"))
# Layer 3: Convolutional. Filters = 48, Filter Size = 5x5, Strides = 2x2.
model.add(Convolution2D(48, 5, strides=(2, 2), activation="elu"))
# Layer 4: Convolutional. Filters = 64, Filter Size = 3x3, Strides = 1x1.
model.add(Convolution2D(64, 3, strides=(1, 1), activation="elu"))
# Layer 5: Convolutional. Filters = 64, Filter Size = 3x3, Strides = 1x1.
model.add(Convolution2D(64, 3, strides=(1, 1), activation="elu"))
# Flatten
model.add(Flatten())
# Layer 6: Fully-connected. Output = 100
model.add(Dense(100, activation="elu"))
# Layer 7: Fully-connected. Output = 50
model.add(Dense(50, activation="elu"))
# Layer 8: Fully-connected. Output = 10
model.add(Dense(10, activation="elu"))
# Layer 9: Fully-connected. Output = 1
model.add(Dense(1, activation="elu"))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=8)

model.save('model.h5')