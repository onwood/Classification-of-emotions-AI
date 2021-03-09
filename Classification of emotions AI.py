import os
import random
from shutil import copyfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, LSTM, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

#------------------------------------------


Training_dir = '../input/emotion-detection-fer/train'
Test_dir = '../input/emotion-detection-fer/test'


#------------------------------------------

train_datagen = ImageDataGenerator(rescale=1 / 255.0,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.4,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(Training_dir,
                                                    target_size=(48, 48),
                                                    class_mode='categorical',
                                                    batch_size=10)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

test_generator = test_datagen.flow_from_directory(Test_dir,
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  batch_size=10)


#------------------------------------------


baseModel = MobileNetV2(weights = 'imagenet', include_top=False, input_shape=(48,48,1))

headModel = baseModel.output
headModel = AveragePooling2D()(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = BatchNormalization()(headModel)
headModel = Dense(64, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = BatchNormalization()(headModel)
headModel = Dense(7, activation='softmax')(headModel)

baseModel.trainable = False
model = Model(inputs = baseModel.input, outputs = headModel)



model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy', metrics=['categorical_accuracy'])


#------------------------------------------


TYPE = 'type'
model_type = 'mobilenetv2'
user = 'block'
iteration = '1'

first_time_training = True

PROJECT_PATH = './'

CHECKPOINT_PATH = PROJECT_PATH + '/checkpoints/' + model_type + '/' + 'by-' + TYPE + '-' + model_type + '-' + user + '-' + iteration + '.h5'
LOGFILE_PATH = PROJECT_PATH + '/log/' + model_type + '/' + model_type + '-by-' + TYPE + '-training-log' + user + '-' + iteration + '.csv'

if not os.path.exists(PROJECT_PATH + '/checkpoints/' + model_type + '/'):
    os.makedirs(PROJECT_PATH + '/checkpoints/' + model_type + '/')

if not os.path.exists(PROJECT_PATH + '/log/' + model_type + '/'):
    os.makedirs(PROJECT_PATH + '/log/' + model_type + '/')

#------------------------------------------


# check point
cp = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                     monitor='val_accuracy',
                     save_best_only=True,
                     verbose=1,
                     mode='auto')

# record log
csv_logger = CSVLogger(filename=LOGFILE_PATH, append=True)


#------------------------------------------


history = model.fit(train_generator,
                     steps_per_epoch = len(train_generator) // 64,
                     validation_data = test_generator,
                     validation_steps = len(test_generator)//64,
                     callbacks = [cp, csv_logger],
                     epochs = 50)


#------------------------------------------



baseModel.trainable = True
fine_tune_at = 100
for layer in baseModel.layers[:fine_tune_at]:
    layer.trainable = False
history2 = model.fit(train_generator,
                     steps_per_epoch = len(train_generator) // 64,
                     validation_data = test_generator,
                     validation_steps = len(test_generator)//64,
                     callbacks = [cp, csv_logger],
                     epochs = 50)



