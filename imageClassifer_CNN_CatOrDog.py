from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.preprocessing import image

import warnings

#cat4 = cv2.imread('../DATA/CATS_DOGS/train/CAT/4.jpg')
#cat4 = cv2.cvtColor(cat4,cv2.COLOR_BGR2RGB)

from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

#plt.imshow(image_gen.random_transform(cat4))

image_gen.flow_from_directory('../DATA/CATS_DOGS/train')
image_gen.flow_from_directory('../DATA/CATS_DOGS/test')

# resizing , width,height,channels
image_shape = (150,150,3)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, remember its binary, 0=cat , 1=dog
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# print(model.summary())

### training the model
batch_size = 16
train_image_gen = image_gen.flow_from_directory('../DATA/CATS_DOGS/train',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')
test_image_gen = image_gen.flow_from_directory('../DATA/CATS_DOGS/test',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')
print(train_image_gen.class_indices)
warnings.filterwarnings('ignore')
results = model.fit_generator(train_image_gen,epochs=100,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                             validation_steps=12)

### Evaluating the Model
print(results.history['acc'])
plt.plot(results.history['acc'])
## Predicting on new images
dog_file = '../DATA/CATS_DOGS/train/Dog/2.jpg'
dog_img = image.load_img(dog_file, target_size=(150, 150))
dog_img = image.img_to_array(dog_img)
dog_img = np.expand_dims(dog_img, axis=0)
dog_img = dog_img/255
prediction_prob = model.predict(dog_img)
print(f'Probability that image is a dog is: {prediction_prob} ')