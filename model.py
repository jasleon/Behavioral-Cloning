import os
import csv
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from math import ceil

def load_data(csv_path, current_path):
    features = ['Center Image', 
                'Left Image', 
                'Right Image', 
                'Steering Angle', 
                'Throttle', 
                'Brake', 
                'Speed']

    # Load data from driving log
    df = pd.read_csv(csv_path, names=features)

    # Read images
    df['Center Image'] = df['Center Image'].apply(lambda x: cv2.imread(current_path + x.split('/')[-1]))
    df['Left Image']   =   df['Left Image'].apply(lambda x: cv2.imread(current_path + x.split('/')[-1]))
    df['Right Image']  =  df['Right Image'].apply(lambda x: cv2.imread(current_path + x.split('/')[-1]))

    # Use multiple cameras
    images = df['Center Image'].tolist()
    measurements = df['Steering Angle'].tolist()
    
    correction = 0.2
    images.extend(df['Left Image'].tolist())
    measurements.extend(df['Steering Angle'].apply(lambda x: x + correction).tolist())

    images.extend(df['Right Image'].tolist())
    measurements.extend(df['Steering Angle'].apply(lambda x: x - correction).tolist())

    # Data augmentation
    images.extend(df['Center Image'].apply(np.fliplr).tolist())
    measurements.extend(df['Steering Angle'].apply(lambda x: -x).tolist())

    images.extend(df['Left Image'].apply(np.fliplr).tolist())
    measurements.extend(df['Steering Angle'].apply(lambda x: -(x + correction)).tolist())

    images.extend(df['Right Image'].apply(np.fliplr).tolist())
    measurements.extend(df['Steering Angle'].apply(lambda x: -(x - correction)).tolist())

    return images, measurements

def load_samples(dnames):
    log = 'driving_log.csv'
    img = 'IMG'
    samples = []
    for dataset in dnames:
        with open(os.path.join(dataset, log)) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                for i in range(3):
                    line[i] = os.path.join(dataset, img, line[i].split('/')[-1])
                samples.append(line)
    return samples

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:(offset + batch_size)]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

samples = load_samples(['training1', 'training2'])
print('samples: {}'.format(len(samples)))
print(samples[0])
print(samples[-1])
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
valid_generator = generator(valid_samples, batch_size=batch_size)

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
model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples)/batch_size),
                    validation_data=valid_generator,
                    validation_steps=ceil(len(valid_samples)/batch_size),
                    epochs=5, verbose=1)

model.save('model.h5')