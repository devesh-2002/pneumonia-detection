# utils.py (inside your app)

import glob
import cv2
import numpy as np

def load_preprocess(path):
    normal = glob.glob(f'{path}/NORMAL/*')
    pneumonia = glob.glob(f'{path}/PNEUMONIA/*')
    X = []
    y = []

    for i in normal:
        img = cv2.imread(i, 0)
        img = cv2.resize(img, (128, 128))
        img = img/255
        img = np.expand_dims(img, axis=-1)  # Add a channel dimension
        X.append(img)
        y.append(0)

    for i in pneumonia:
        img = cv2.imread(i, 0)
        img = cv2.resize(img, (128, 128))
        img = img/255
        img = np.expand_dims(img, axis=-1)  # Add a channel dimension
        X.append(img)
        y.append(1)

    return X, y
