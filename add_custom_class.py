# This script adds custom classes to open-source image recognition models.

# Data: I use the "Dogs vs. Cats" dataset from Kaggle.com.
# https://www.kaggle.com/c/dogs-vs-cats
# For each class, I use 100 images for both the training and validation sets.

# Strategy: use everything from open-source image recognition models EXCEPT
# the last layer. Run the new training data through this almost-full network.
# Train a small, fully-connected neural net on the outputs of the almost-full model.
# More info here: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# And here: https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# And here: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

################
# IMPORT MODULES
################
import os
import numpy as np
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import xception, inception_v3

##########################
# SPECIFY MODEL PARAMETERS
##########################

batch_size = 10           # Specify batch size.
training_images = 200     # Specify number of training images.
val_images = 200          # Specify number of validation set images.
target_size_1 = 299       # Specify dimensions of target image
target_size_2 = 299
num_classes = 2           # Specify number of classes
epochs = 1                # Specify number of training epochs.

#################################
# CREATE IMAGE RECOGNITION MODELS
#################################
# This section creates image recognition model objects.
# These models need to use weights derived from training
# on the ImageNet data set (as opposed to random weights,
# which would probably suck).

# The weights are too large to be uploaded to Github and so,
# are NOT in the 'weights' folder initially. They can be downloaded from:
# Xception: https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5
# InceptionV3: https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5

# For this script to work, the weights must be in the 'weights' folder
# and named as follows:
# Xception weights file: xception_weights_tf_dim_ordering_tf_kernels.h5
# InceptionV3 file: inception_v3_weights_tf_dim_ordering_tf_kernels.h5

# Create models (make sure weights are already downloaded!).
# Make sure weights are already downloaded!
# Make sure working directory is the repository!
xc = xception.Xception(weights = 'weights/xception_weights_tf_dim_ordering_tf_kernels.h5')
ic3 = inception_v3.InceptionV3(weights = 'weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

# Remove top layers.
xc.layers.pop()
ic3.layers.pop()

##############################################
# RUN TRAINING DATA THROUGH ALMOST-FULL MODELS
##############################################
# This section runs training data through the almost-full image recognition models.
# The resulting output is used as input to a small fully-connected network later on.

# Create image data generator.
generator = image.ImageDataGenerator(rescale = 1. / 255)

# Create batch generator (which uses image generator).
# I want to keep the data in order so I can keep track of labels ("shuffle = False").
train_batch_gen = generator.flow_from_directory('data/train',
                                                target_size = (target_size_1, target_size_2),
                                                batch_size = batch_size,
                                                class_mode = 'categorical',
                                                shuffle = False)

# Get features from training data for almost-full image rec models.
xc_train_features = xc.predict_generator(train_batch_gen,
                                         (training_images / batch_size))
ic3_train_features = ic3.predict_generator(train_batch_gen,
                                           (training_images / batch_size))

# Save training labels.
train_labels = train_batch_gen.classes

# Repeat the process (generating features) for validation data.

# Create batch generator (which uses image generator).
# I want to keep the data in order so I can keep track of labels ("shuffle = False").
cv_batch_gen = generator.flow_from_directory('data/validation',
                                             target_size = (target_size_1, target_size_2),
                                             batch_size = batch_size,
                                             class_mode = 'categorical',
                                             shuffle = False)

# Get features from validation data.
xc_val_features = xc.predict_generator(cv_batch_gen,
                                       (val_images / batch_size))
ic3_val_features = xc.predict_generator(cv_batch_gen,
                                        (val_images / batch_size))

# Save validation set labels.
val_labels = cv_batch_gen.classes

print('All Features Generated')
print('')

#################################
# TRAIN MODEL ON DERIVED FEATURES
#################################
# I now have features generated from running images through the almost-full image rec models.
# This section uses those features to train a small neural network.

# Convert training and validation set labels to one-hot format.
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)

# Create first model.
xc_model = Sequential()                                        # create sequential model
xc_model.add(Dense(256,                                        # add dense layer
                input_shape = xc_train_features.shape[1:],
                activation = 'relu'))                  
xc_model.add(Dropout(0.5))                                     # add dropout
xc_model.add(Dense(num_classes, activation = 'sigmoid'))       # For >= 3 classes, use "activation = 'softmax'"
                                                            # See "https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/" for reference

# Configure model.
xc_model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',            # For >= 3 classes, use "loss = 'categorical_crossentropy'"
                                                       # See "https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/" for reference
              metrics = ['accuracy'])

# Create second model.
ic3_model = Sequential()                                        # create sequential model
ic3_model.add(Dense(256,                                        # add dense layer
                input_shape = ic3_train_features.shape[1:],
                activation = 'relu'))                  
ic3_model.add(Dropout(0.5))                                     # add dropout
ic3_model.add(Dense(num_classes, activation = 'sigmoid'))       # For >= 3 classes, use "activation = 'softmax'"
                                                            # See "https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/" for reference

# Configure model.
ic3_model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',            # For >= 3 classes, use "loss = 'categorical_crossentropy'"
                                                       # See "https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/" for reference
              metrics = ['accuracy'])

# Fit both models.
np.random.seed(20180405)
xc_model.fit(xc_train_features,
          train_labels,
          epochs = epochs,
          batch_size = batch_size,
          validation_data = (xc_val_features, val_labels),
          verbose = 2,
          shuffle = False)

np.random.seed(20180405)
ic3_model.fit(ic3_train_features,
          train_labels,
          epochs = epochs,
          batch_size = batch_size,
          validation_data = (ic3_val_features, val_labels),
          verbose = 2,
          shuffle = False)

##################
# EVALUATE RESULTS
##################

# View some of the labels.
val_labels[0:9]

# View some predicted probabilities for validation set images.
xc_model.predict(xc_val_features[0:9])
ic3_model.predict(ic3_val_features[0:9])

# Evaluate both models on entire validation set.
# Eh, not that great...
xc_model.evaluate(xc_val_features, val_labels)
ic3_model.evaluate(ic3_val_features, val_labels)

# Evaluate accuracy of both models per class.

# cats
xc_model.evaluate(xc_val_features[0:99], val_labels[0:99])
ic3_model.evaluate(ic3_val_features[0:99], val_labels[0:99])

# dogs
xc_model.evaluate(xc_val_features[100:199], val_labels[100:199])
ic3_model.evaluate(ic3_val_features[100:199], val_labels[100:199])

##################
# EXPLORE MISTAKES
##################
# This section only really makes sense if there are =>3 classes.
# In that case, it may be useful to examine, for example,
# when the model gets a category 1 image wrong, does it think the image
# is in category 2 or category 3?

# Get predictions for cat images.
xc_cat_preds = np.argmax(xc_model.predict(xc_val_features[0:99]), axis = 1)
ic3_cat_preds = np.argmax(ic3_model.predict(ic3_val_features[0:99]), axis = 1)

# Look at how many predictions fall into each class (trivial for the 2 class case).
(xc_cat_preds == 0).sum() / len(xc_cat_preds)
(xc_cat_preds == 1).sum() / len(xc_cat_preds)

(ic3_cat_preds == 0).sum() / len(ic3_cat_preds)
(ic3_cat_preds == 1).sum() / len(ic3_cat_preds)

# Get predictions for dog images.
xc_dog_preds = np.argmax(xc_model.predict(xc_val_features[100:199]), axis = 1)
ic3_dog_preds = np.argmax(ic3_model.predict(ic3_val_features[100:199]), axis = 1)

# Look at how many predictions fall into each class (trivial for the 2 class case).
(xc_dog_preds == 0).sum() / len(xc_dog_preds)
(xc_dog_preds == 1).sum() / len(xc_dog_preds)

(ic3_dog_preds == 0).sum() / len(ic3_dog_preds)
(ic3_dog_preds == 1).sum() / len(ic3_dog_preds)
