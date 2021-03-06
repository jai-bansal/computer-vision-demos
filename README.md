## computer-vision-demos

### Summary:
This repo contains simple examples of a few popular computer vision tasks:
- image classification using transfer learning
- object character recognition (OCR)
- object detection

There's a branch for each task.

### Data: 

#### Image Classification using Transfer Learning
Data comes from the "Dogs vs. Cats" dataset from Kaggle.com.
To keep things easy, I only use 100 images for the training and validation sets for each class.

The data can be found at: https://www.kaggle.com/c/dogs-vs-cats

#### OCR
I use 3 images containing text (included in repository).

#### Object Detection: 
To train the object detection model, I downloaded ~175 bear pictures from Google Images.
To qualitatively test the model, I downloaded a video with bears in it from Youtube.

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
I attempt to identify bears in a new, test video and place bounding boxes around bears whenever they appear.

I generally use a transfer learning approach similar to the "image_classification_transfer_learning" branch. I start with a pre-trained model and train it further using ~175 bear images from Google Images.

Available models can be found here:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Reference articles:
- https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95
- https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

I especially used the first post above. The scripts in this repo (and the first post above) are basically modified versions of scripts found in the Tensorflow Object Detection API repo (https://github.com/tensorflow/models/tree/master/research/object_detection).

I run these scripts locally, not in the cloud. Training the model to decent performance took ~2 days using only a CPU.

Note: getting the scripts running can be tricky. You should have the following repositories cloned:
- https://github.com/tensorflow/models
- https://github.com/cocodataset/cocoapi

For the 2nd repo above, if you are on Windows I recommend using:
- https://github.com/philferriere/cocoapi

You'll also need to install Visual C++ (instructions in the link above).

Here are instructions for installing the Tensorflow Object Detection API: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

These instructions require installing "protoc". I recommend using "protoc-3.4.0-win32.zip" which can be found here: 
https://github.com/google/protobuf/releases/tag/v3.4.0

The newest versions of "protoc" seemed to be missing a file I needed and did not work for me. Note that I'm using a Windows machine.

Even with the above instructions, I hit many errors due to not setting my Python path correctly and not compiling Protobuf correctly. The Tensorflow Object Detection API is constantly changing, so things may be out of date!

The final product ('test_video/bear sits next to guy_detection.mp4') is pretty good but not perfect. The model seems to mistake anything brown for a bear (like the camping chair at 0:02). It doesn't detect the actual bear until about 0:05. It stops recognizing the bear when the bear turns its face away from the camera at 0:08. This is probably a result of most of the training data being pictures of bears from the front. There are other false positives, but these might be excluded by only including detection above some confidence threshold. Overall, this is a pretty cool result.

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
- 'add_custom_class.py': trains a model to identify custom classes (dogs and cats)

#### 'optical_character_recognition' branch

- 'sample_images' folder: contains 3 sample images to conduct OCR on
- 'ocr.R' and 'ocr.py': conducts OCR (returning text and images with bounding boxes) in R and Python respectively

#### 'object_detection' branch

- 'annotations' folder: contains annotations for images in the 'images' folder. I used an annotation tool (https://github.com/tzutalin/labelImg). This folder also contains 'trainval.txt' (output of 'scripts/trainval_txt_creator.py').
- 'base_model' folder: holds pre-trained model files (empty in repo because files are too large). I used 'ssd_mobilenet_v1_coco' from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
- 'eval_dir' folder: contains model evaluation artifacts, specifically a 'tfevents' file (not in repo because files are too large). This is the output of 'scripts/eval.py'. The folder does contain a video ('sample_pic_eval_over_time.mp4') of the model being evaluated on a sample image at different training iterations. It's cool to see the model improve and eventually be able to identify the bear in the image.
- 'images' folder: contains ~175 bear images used to train object detection model. I manually downloaded these from Google Images.
- 'inference_ready_model': holds trained object detection model ready for use on new images (empty in repo because files are too large). These files are the output of 'scripts/export_inference_graph.py'.
- 'prepped_data' folder: contains training and validation set data prepared for use in model training. These files are outputs of 'scripts/create_record.py')
- 'scripts' folder:
   - 'create_record.py': creates training and validation data set records used to train an object detection model
   - 'eval.py': evaluates a trained object detection model
   - 'export_inference_graph.py': takes a trained object detection model as input and outputs a version of that model ready for use on new images
   - 'run_model_on_new_video.py': runs an object detection model on a new video (technically the frames of the video)
   - 'train.py': trains an object detection model. Takes as input 'prepped_data/train.record' and 'prepped_data/val.record'.
   - 'trainval_txt_creator.py': creates 'annotations/trainval.txt'. This file matches images with bear labels and seems to be necessary for the whole process to work.
- 'test_video' folder: contains a video that trained object detection model is qualitatively tested on and the object-detected version of that video.
- 'training_artifacts' folder: contains model training artifacts, specifically recent model checkpoints (empty in repo because files are too large). These are outputs of running 'scripts/train.py'.
- 'label_map.pbtxt': lists the classes, in this case just 'bear'
- 'ssd_mobilenet_v1_coco_2018_01_28.config': config file for the model I used

- Here's a recommended order to work through the scripts: 'trainval_txt_creator.py', 'create_record.py', 'train.py', 'eval.py', 'export_inference_graph.py', 'run_model_on_new_video.py'. I also recommend using the reference articles linked above.
