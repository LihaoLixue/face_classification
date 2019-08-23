import cv2
import numpy as np
import pandas as pd
from pandas import read_csv

data = read_csv('../datasets/fer2013/fer2013.csv')
print(data.head())
print(data.shape)
dataset = data.values
image_size=(48, 48)

pixels = data['pixels'].tolist()
width, height = 48, 48
faces = []
# for pixel_sequence in pixels:
#     face = [int(pixel) for pixel in pixel_sequence.split(' ')]
#     face = np.asarray(face).reshape(width, height)
#     face = cv2.resize(face.astype('uint8'), image_size)
#     faces.append(face.astype('float32'))
# faces = np.asarray(faces)
# faces = np.expand_dims(faces, -1)
emotions = pd.get_dummies(data['emotion']).as_matrix()
print(emotions)
# return faces, emotions
