import glob
import cv2
from django.shortcuts import render
import numpy as np
from tensorflow import keras
def my_view(request):
    return render(request, 'index.html')

from django.shortcuts import render
from .utils import load_preprocess


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

# views.py

import os
import cv2
from django.shortcuts import render
def predict(request):
    if request.method == 'POST':
        file = request.FILES['img']
        file_path = os.path.join('uploads', file.name)

        with open(file_path, 'wb') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        model = keras.models.load_model('models/pneumonia.h5')
        
        test_image_read = cv2.imread(file_path, 0)  # Read as grayscale
        test_image = cv2.resize(test_image_read, (128, 128))
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=-1)
        # Make predictions using the model
        prediction = model.predict(np.expand_dims(test_image, axis=0))
        print((prediction))
        if(np.argmax(prediction)==1):
            print ("Pneumonia Detected")
        else : 
           print ("Normal")

        return render(request, 'result.html', {'prediction_text': prediction, 'image_file_name': file.name})

    return render(request, 'index.html')