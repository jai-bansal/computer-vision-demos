## computer-vision-demos

### Summary:
This repo contains simple examples of a few popular computer vision tasks:
- image classification using open source image recognition models modified for custom classes
- object character recognition (OCR) (TO DO)
- object detection (TO DO)

### Data: 
Data comes from the "Dogs vs. Cats" dataset from Kaggle.com.
To keep things easy, I only use 100 images for the training and validation sets for each class.

The data can be found at: https://www.kaggle.com/c/dogs-vs-cats

### Method:

#### Image Classification with Custom Classes
I attempt to identify 2 custom classes: cats and dogs.

The general strategy is to use an open-source image recognition model without the last layer.
I run the new data through this almost-complete model and use the outputs as the input for a 
new, small neural network.

Reference articles:
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html\
- https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
- https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

I use 2 open source image recogntion models:
- Xception
- InceptionV3

No parameter tuning is included, this repo just gets things running as fast as possible.

#### OCR

#### Object Detection

### Files

- 'data' folder
   - 'train' folder
      - 'cat' folder: contains training images of cats
      - 'dog' folder: contains training images of dogs
   - 'validation' folder
      - 'cat' folder: contains validation set images of cats
      - 'dog' folder: contains validation set images of dogs
- 'weights' folder: initially just contains a placeholder text file but must contain the weights file for the image rec models above for the script to run
- 'add_custom_class.py': trains a model to identify custom classes of dogs and cats
