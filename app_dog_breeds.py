import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
from glob import glob
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image

# custom functions
from extract_bottleneck_features import *


@st.cache
def init_resnet50():
    # define ResNet50 model

    ResNet50_model = ResNet50(weights='imagenet')
    return ResNet50_model


@st.cache
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


@st.cache
def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path

    ResNet50_model = init_resnet50()
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

@st.cache
def dog_detector(img_path):
    # returns true if dog is detected in the passed image

    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

@st.cache
def face_detector(img_path):

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


@st.cache
def Inception_predict_breed(img_path):
    # Predict the breed of dog using Inceptionv3 pre-trained network
    
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Inception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


@st.cache
def predict_dog_breed(img_path):
    # predict the breed of the dog based on input image
    
    dog_detected = dog_detector(img_path)
    human_detected = face_detector(img_path)
    image = Image.open(img_path)
    if dog_detected or human_detected:
        dog_breed = Inception_predict_breed(img_path)
        dog_breed = dog_breed[dog_breed.find('.')+1:]
        dog_breed = dog_breed.replace('_'," ")
        message_string = ('The photo is of a ' + dog_breed) if dog_detected\
                         else ('The face in the photo looks like a ' + dog_breed)
        # print(message_string)
    else:
        message_string = "Error: The image doesn't seem to be that of a human or a dog. Please try a different image" 
        # print(message_string)
    return message_string

def get_subfolder_names():
    # function to get all folder names in the app folder

    cwd = os.getcwd()
    dir_list = next(os.walk(cwd))[1]
    return dir_list


def get_file_names(subfolder_name):
    # get all files in the selected subfolder

    cwd = os.getcwd()
    file_list = next(os.walk(cwd + "\\" + subfolder_name))[2]
    return file_list


def main():

    # title of the app
    st.title('Dog Breed Classifier')

    # select the folder in which the test images are stored
    folder_select =\
    st.sidebar.selectbox('Select the folder name where the images are stored',
                         get_subfolder_names())

    # search the test_images folder for input images
    file_select =\
    st.sidebar.selectbox('Select the image file name',
                         get_file_names(folder_select))

    # build image path
    cwd = os.getcwd()
    img_path = cwd + "\\" + folder_select + "\\" + file_select

    try:
        # display selected image file
        img = Image.open(img_path)
        st.image(img, use_column_width=True)
    except IOError:
        st.write('Please provide a valid image file')

    try:
        # predict dog breed and display text
        st.write(predict_dog_breed(img_path))
    except Exception as e:
        st.write('Prediction failed. Please try again', e)

if __name__ == "__main__":
    main()