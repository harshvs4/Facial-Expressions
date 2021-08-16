import numpy as numpy
import cv2
import tensorflow 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

#Train directory
train_dir = 'data/train'

#Validate directory
val_dir = 'data/test'

#Generating images
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (48,48), #Target size(in pixels)
    batch_size = 64, #Batch size in machine learning is the number of examples utilized in one iteration
    color_mode = "grayscale",
    class_mode = 'categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
)

#Create model and prepare it for training

emotion_model = Sequential() #Sequential allows you to prepare model layer by layer

#Adding layers
emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax')) # 7 categories(check test database)[1024 categories get converted into a total of 7 categories]

#Compile,train and save the model

#loss=degree of error(models try to minimze the loss but don't try to maximize the accuracy)
emotion_model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.0001, decay=1e-6),metrics=['Accuracy'])

emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch = 28709 // 64,
    epochs = 50, #epochs = number of iterations
    validation_data = validation_generator,
    validation_steps = 7178 // 64
)

emotion_model.save_weights('model.h5')