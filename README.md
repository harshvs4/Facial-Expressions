# Facial-Expressions
Detecting facial expressions using Python and Tensorflow

## Objective
Aim of the project is to build and train a convolutional neural network (CNN) in Keras from scratch to recognize facial expressions.The objective is to classify each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).OpenCV is used to automatically detect faces in images and draw bounding boxes around them.
This code converts the facial expressions into emojis from one of the seven categories

## Environment

* Anaconda Command Prompt
* Visual Studio Code

## Packages used
```TensorFlow```
```Keras```
```Numpy```
```OpenCV```
```TkInter```

# Datatset
[Kaggle](https://www.kaggle.com/msambare/fer2013) FER-2013

## TensorFlow Installation

* Open Anaconda Navigator
* Create an Environment and name it **tensorflow**
* Install the ```TensorFlow``` package(2.4.0)
* After ```TensorFlow``` is installed, install the ```Keras``` and the ```OpenCV``` package
* Install the Command prompt for the **tensorflow** environment and use it to train and run the models

## ImageMagicK Installation

[ImageMagicK](https://imagemagick.org/script/download.php)

## Steps

* First train the model(train.py) in the Anaconda command prompt

``` python train.py```

* Run the program

``` python emoji.py```
