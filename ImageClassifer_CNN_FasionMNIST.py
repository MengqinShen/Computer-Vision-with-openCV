from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)


#plt.imshow(x_train[0])

## Preprocessing the Data

# Normalize the X train and X test data by dividing by the max value of the image arrays
x_train = x_train/x_train.max()
x_test = x_test/x_train.max()
row,col,channel  = x_train.shape
x_train = x_train.reshape(row,col,channel,1)
x_test = x_test.reshape(row/6,col,channel,1)

## Use Keras to create a model consisting of at least the following layers
model = Sequential()
# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

## Train/Fit the model to the x_train set
model.fit(x_train,y_cat_train,epochs=10)

## Evaluating the Model
#model.metrics_names
print(model.evaluate(x_test,y_cat_test))
predictions = model.predict_classes(x_test)
print(predictions)
print(classification_report(y_test,predictions))