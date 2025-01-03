# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:45:20 2023

@author: prake
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense



img_width, img_height = 150, 150

train_data_dir = r"C:\Users\prake\Desktop\AIML\spy_DL\dogcat\train"
validation_data_dir = r"C:\Users\prake\Desktop\AIML\spy_DL\dogcat\validation"
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 3


#input layer
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#hidden layer-1
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#output layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=32,
        epochs=13,
        validation_data=validation_generator)

#This part helps the trained model to classify the output class...
import numpy as np
from keras.preprocessing import image
test_image=image.load_img(r'C:\Users\prake\Desktop\AIML\spy_DL\dogcat\train\dogs\dog.2.jpg',target_size=(150,150))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
validation_generator.class_indices
if result[0][0]>=0.5:
    print('dog')
else:
    print('cat')

# To save the model with some name ....
from keras.models import model_from_json
from keras.models import load_model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")















