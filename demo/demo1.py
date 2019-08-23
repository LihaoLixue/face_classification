import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 2),
                 input_shape=(8,8,1)))
convout1 = Activation('relu')
model.add(convout1)

model.add(Conv2D(64, (2, 3), activation='relu'))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

