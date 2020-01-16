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
from keras.models import model_from_json
import pickle

# custom functions
from extract_bottleneck_features import *


# @st.cache
def init_resnet50():
    """
    Initialize ResNet50 model

    Parameters
    ----------
    The function doesn't take any parameters

    Returns
    -------
    ResNet50 model
        ResNet50 Keras model
    """

    ResNet50_model = ResNet50(weights='imagenet')
    return ResNet50_model


# @st.cache
def path_to_tensor(img_path):
    """
    Convert image file (specified as path to its location) into a tensor,
    also make the dimensions consistent for use with TensorFlow library
    and training DL models

    Parameters
    ----------
    img_path
        Path to an image file that has to be converted

    Returns
    -------
    numpy 4-d array
        A numpy 4d array (tensor) that can be used with TensorFlow
        with dimensions of 1 x 224 x 224 x 3
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return
    return np.expand_dims(x, axis=0)


# @st.cache
def ResNet50_predict_labels(img_path):
    """
    Returns prediction of image category based on ResNet50 model trained on
    the ImageNet dataset.

    Parameters
    ----------
    img_path
        Path to an image file for which the category has to be predicted

    Returns
    -------
    integer
        Category classification of the provided image
    """

    ResNet50_model = init_resnet50()
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


# @st.cache
def dog_detector(img_path):
    """
    Returns True if dog is detected in the specified image and False otherwise.
    Uses ResNet50 model trained on the ImageNet dataset for this.

    Parameters
    ----------
    img_path
        Path to the image file

    Returns
    -------
    boolean
        True if dog is detected in the passed image else False
    """

    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


# @st.cache
def face_detector(img_path):
    """
    Returns True if dog is detected in the specified image and False otherwise.
    Uses ResNet50 model trained on the ImageNet dataset for this.

    Parameters
    ----------
    img_path
        Path to the image file

    Returns
    -------
    boolean
        True if dog is detected in the passed image else False
    """
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# @st.cache
def Inception_predict_breed(img_path, Inception_model, dog_names):
    """
    Predict the breed of dog using Inceptionv3 pre-trained model

    Parameters
    ----------
    img_path
        Path to the image file

    Inception_model
        Pretrained Inception model (TensorFlow/Keras base)

    dog_names
        Mapping for numerical dog categories that the model predicts
        to a string denoting the name of the dog for output

    Returns
    -------
    string
        Predicted name of the dog breed based on the specified image
    """

    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))

    # obtain predicted vector
    predicted_vector = Inception_model.predict(bottleneck_feature)

    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


# @st.cache
def predict_dog_breed(img_path, Inception_model, dog_names):
    """
    Predict the breed of the dog based on input image and model (Inceptionv3).
    If the input image is detected as a human face or a dog,
    the output is the closest dog breed.
    If the input image is detected as neither a dog or a human face,
    the output message asks the user to select a different image 

    Parameters
    ----------
    img_path
        Path to the image file

    Inception_model
        Pretrained Inception model (TensorFlow/Keras base)

    dog_names
        Dictionary mapping for numerical dog categories that the model predicts
        to a string denoting the name of the dog for output

    Returns
    -------
    string
        A string with the message that has to be printed out to the app user.
        Specifies if the input image is a face and which dog breed it looks like.
        Alternately, if a dog is detected, specifies the dog breed.
        If the specified image isn't a dog or a human, the message asks the user
        to select a different image
    """

    # check if input image is dog/human face
    dog_detected = dog_detector(img_path)
    human_detected = face_detector(img_path)
    image = Image.open(img_path)

    # perform prediction only if image is detected as dog or face
    if dog_detected or human_detected:
        dog_breed = Inception_predict_breed(img_path, Inception_model, dog_names)
        dog_breed = dog_breed[dog_breed.find('.')+1:]
        dog_breed = dog_breed.replace('_', " ")
        message_string = ('The photo is of a ' + dog_breed) if dog_detected\
                         else ('The face in the photo looks like a ' + dog_breed)
        # print(message_string)

    # ask user to select different image if not dog or human face
    else:
        message_string = "Error: The image doesn't seem to be that of a human or a dog. Please try a different image" 
        # print(message_string)
    return message_string


def get_subfolder_names():
    """
    Get a list of all subfolders in the root directory of the py file.

    Parameters
    ----------
        The function doesn't take any parameters

    Returns
    -------
    list of strings
        A list of all subfolders in the directory of the app.
    """
    # function to get all folder names in the app folder

    cwd = os.getcwd()
    dir_list = next(os.walk(cwd))[1]
    return dir_list


def get_file_names(subfolder_name):
    """
    Get a list of all files in the specified input directory.

    Parameters
    ----------
    subfolder_name
        String specifying path to subfolder

    Returns
    -------
    list of strings
        A list of all files in the input directory.
    """
    # get all files in the selected subfolder

    cwd = os.getcwd()
    file_list = next(os.walk(cwd + "\\" + subfolder_name))[2]
    return file_list


def load_model_labels():
    """
    Predict the breed of the dog based on input image and model (Inceptionv3).
    If the input image is detected as a human face or a dog,
    the output is the closest dog breed.
    If the input image is detected as neither a dog or a human face,
    the output message asks the user to select a different image

    Parameters
    ----------
    img_path
        Path to the image file

    Inception_model
        Pretrained Inception model (TensorFlow/Keras base)

    dog_names
        Dictionary mapping for numerical dog categories that the model predicts
        to a string denoting the name of the dog for output

    Returns
    -------
    A tuple containing the dog name mapping and Inceptionv3 model
    dictionary
        A dictionary with mapping of the numeric categories of dog breeds
        to the text label associated with it.
    model
        The trained Inceptionv3 model
    """

    # Load dog breed labels
    with open('dog_names.pkl', 'rb') as dog_names_file:
        dog_names = pickle.load(dog_names_file)

    # Import model
    with open('saved_models\inception_model.json', 'rb') as model_json_file:
        Inception_model = model_from_json(model_json_file.read())

    # Load model weights
    Inception_model.load_weights('saved_models\weights.best.Inception.hdf5')

    return dog_names, Inception_model


def main():

    # output title of the app
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
        # resize and display selected image file
        img = Image.open(img_path)
        size = 600, 500
        img.thumbnail(size, Image.ANTIALIAS)
        st.image(img)
    
    except IOError:
        # output error message if file not found
        st.write('Please provide a valid image file')

    try:
        # load model and dog labels
        dog_names, Inception_model = load_model_labels()
    except Exception as e:
        print("Error: Couldn't find required files", e)

    try:
        # predict dog breed and display text
        st.write(predict_dog_breed(img_path, Inception_model, dog_names))
    except Exception as e:
        st.write('Prediction failed. Please try again', e)


if __name__ == "__main__":
    main()
    