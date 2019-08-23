from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

data_iter = datagen.flow(x_train, y_train, batch_size=8)

while True:
    x_batch, y_batch = data_iter.next()
    for i in range(8):
        print(i // 4)
        plt.subplot(2, 4, i + 1)
        plt.imshow(x_batch[i].reshape(28, 28), cmap='gray')
    plt.show()