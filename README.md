# Facial-Gender-classification
Basic Convolutional Neural network is implemented for classifying Gender of persons from input images or videos using OpenCV with python.


#Dataset
Dataset has been taken from kaggle url:-https://www.kaggle.com/gmlmrinalini/genderdetectionface? which consist of two categories man and woman each with 800 training images and 170 test images i.e. total training images=1600 and total test images= 340

#Model
layers : Conv2D--->MaxPool2D--->Dropout---->Conv2D--->MaxPool2D--->Dropout---->Conv2D--->MaxPool2D--->Dropout---->Flatten--->Dense---->Dropout--->Dense
activation : (relu) (relu) (relu) (relu) (sigmoid)
filter : 32,(3,3), 64,(3,3)
pool_size = (2,2)
input_shape = (64,64,1)
output_shape = 128

#Metrics
loss='binary_crossentropy'
optimizer='adam'
metrics=['accuracy']

#Model format
.h5

#How to run ?

  Files:-
  mainimage.py:- code for classifying gender using image input.
  mainvideo.py:- code for classifying gender using video input from webcam.
  haarcascade_frontalface_alt.xml:- Used for detecting faces which is used as an input to our model.
  Detectfacialgender.ipynb:- CNN code for training network and storing in .h5 format i.e.  "detectfacemodel.h5".

  note:-rest contains image file.

1. For classifying gender from images run command "python mainimage.py"
2. For classifying gender from webcam run command "python mainvideo.py

