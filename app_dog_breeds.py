import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2

# custom functions
from extract_bottleneck_features import *

def face_detector(img_path):

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def Inception_predict_breed(img_path):
    # Predict the breed of dog using Inceptionv3 pre-trained network
    
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Inception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def predict_dog_breed(img_path):
    # 
    
    dog_detected = dog_detector(img_path)
    human_detected = face_detector(img_path)
    image = mpimg.imread(img_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    if dog_detected or human_detected:
        dog_breed = Inception_predict_breed(img_path)
        dog_breed = dog_breed[dog_breed.find('.')+1:]
        dog_breed = dog_breed.replace('_'," ")
        message_string = ('The photo is of a ' + dog_breed) if dog_detected\
                         else ('The face in the photo looks like a ' + dog_breed)
        print(message_string)
    else:
        print("Error: The image doesn't seem to be that of a human or a dog. Please try a different image")