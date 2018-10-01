## computer-vision-demos

### Summary:
This repo contains simple examples of a few popular computer vision tasks:
- image classification using transfer learning
- object character recognition (OCR)
- object detection (IN PROGRESS)

### Data: 

#### Image Classification using Transfer Learning
Data comes from the "Dogs vs. Cats" dataset from Kaggle.com.
To keep things easy, I only use 100 images for the training and validation sets for each class.

The data can be found at: https://www.kaggle.com/c/dogs-vs-cats

#### OCR
I use 3 images containing text (included in repository).

#### Object Detection: 
To train the object detection model, I downloaded ~175 bear pictures from Google Images.
To qualitatively test the object detection model, I downloaded a video with bears in it from Youtube.

### Method:

#### Image Classification using Transfer Learning
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
I conduct OCR on 3 sample images in R and Python. I return text and the original images with bounding boxes.

#### Object Detection: 
I attempt to identify a bear in a new, test video and place a bounding box around bears in every frame in which they appear.

I generally use a transfer learning approach similar to the "image_classification_transfer_learning" branch. I start with a pre-trained network and train it using my custom bear images.

Much of my approach was based on this blog post:
https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95

I run these scripts locally, not in the cloud.

Note: getting the object detection scripts running is pretty tricky. You should have the following repositories cloned:
- https://github.com/tensorflow/models
- https://github.com/cocodataset/cocoapi

Here are helpful instructions for the installation of the Tensorflow Object Detection API: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

These instructions require installing "protoc". I recommend using "protoc-3.4.0-win32.zip" which can be found here: 
https://github.com/google/protobuf/releases/tag/v3.4.0

The newest versions of "protoc" seemed to be missing a file I needed and did not work for me.

Please follow the instructions in the above link carefully...I hit many errors due to not setting my Python path correctly and not compiling Protobuf correctly.

Available models can be found:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

You will probably still hit a few errors that need to be Googled.

### Branches and Files

#### 'image_classification_transfer_learning' branch

- 'data' folder
   - 'train' folder
      - 'cat' folder: contains training images of cats
      - 'dog' folder: contains training images of dogs
   - 'validation' folder
      - 'cat' folder: contains validation set images of cats
      - 'dog' folder: contains validation set images of dogs
- 'weights' folder: initially just contains a placeholder text file but must contain the weights file for the image rec models above for the script to run
- 'add_custom_class.py': trains a model to identify custom classes of dogs and cats

#### 'optical_character_recognition' branch

- 'sample_images' folder: contains 3 sample images to conduct OCR on
- 'ocr.R' and 'ocr.py': conducts OCR (returning text and images with bounding boxes) in R and Python respectively

#### 'object_detection' branch

- 'annotations' folder: contains annotations for the images in the 'images' folder. I used a tool to create annotations (https://github.com/tzutalin/labelImg). This folder also contains 'trainval.txt' (output of running 'scripts/trainval_txt_creator.py').
- 'base_model' folder: holds pre-trained model files (empty in repo because files are too large). I used 'ssd_mobilenet_v1_coco' from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
- 'eval_dir' folder: contains model evaluation artifacts, specifically a 'tfevents' file (empty in repo because files are too large). These are outputs of running 'scripts/eval.py'.
- 'images' folder: contains images used to train object detection model
- 'prepped_data' folder: contains training and validation set data prepared for use in model training (these are outputs of running 'scripts/create_record.py')
- 'scripts' folder: contains all scripts for this project. All scripts have descriptions in their comments.
- 'test_video' folder: contains videos that trained object detection model is qualitatively tested on
- 'test_video_frames' folder: contains frames from the videos in 'test_video' folder (empty in repo because files are too large). These are outputs of running 'scripts/split_vids_into_frames.py'
- 'training_artifacts' folder: contains model training artifacts, specifically recent model checkpoints (empty in repo because files are too large). These are outputs of running 'scripts/train.py'.
