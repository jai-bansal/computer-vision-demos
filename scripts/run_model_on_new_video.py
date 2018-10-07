# This script uses a trained object detection model on a new video.
# This script is bassed on a modified version of this Jupyter notebook:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# I add code using the "moviepy" package to work with videos.

################
# IMPORT MODULES
################

import numpy as np                                   
import os                                            
import tensorflow as tf                              

from PIL import Image                                 

from utils import label_map_util                                         
from utils import visualization_utils as vis_util  

import cv2                                           # Deals with images.
from moviepy.editor import *                         # Deals with video clips

##################
# DEFINE VARIABLES
##################

PATH_TO_FROZEN_GRAPH = "inference_ready_model/frozen_inference_graph.pb"               # Model checkpoint exported for inference.
PATH_TO_TEST_IMAGES_DIR = "test_video_frames"                                          # Location of test video frames.
PATH_TO_LABELS = 'label_map.pbtxt'
TEST_IMAGE_PATHS = ["test_video_frames/bear sits next to guy_" + str(i) + ".0.jpg" for i in range(1, len(os.listdir("test_video_frames")) + 1)]

###############################
# LOAD FROZEN MODEL INTO MEMORY
###############################

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name = '')
    
################
# LOAD LABEL MAP
################

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

#################################################
# DEFINE FUNCTION TO RUN DETECTION ON A NEW IMAGE
#################################################
  
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

###############################################
# RUN TEST VIDEO THROUGH OBJECT DETECTION MODEL
###############################################
# This section runs the frames of the test video through the object detection 
# model, saves the object-detected frames, and turns them into a new video.
  
# Import original version of video.
orig_vid = VideoFileClip("test_video/bear sits next to guy.mp4")

# Create list to hold object detected frames.
detected_frames = []

# Loop over video frames.
for frame in orig_vid.iter_frames():
    
  # Keep track of progress.
  if len(detected_frames) > 0 and len(detected_frames) % 100 == 0:
      print('Frames Done: ', len(detected_frames))
  
  # Actual detection.
  output_dict = run_inference_for_single_image(frame, detection_graph)
  
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(frame,
                                                     output_dict['detection_boxes'],
                                                     output_dict['detection_classes'],
                                                     output_dict['detection_scores'],
                                                     category_index,
                                                     instance_masks = output_dict.get('detection_masks'),
                                                     use_normalized_coordinates = True,
                                                     line_thickness = 8)
  
  # Add object-detected frame to "detected_frames" list.
  detected_frames.append(frame)
  
# Turn "detected_frames" into a video and export.
new_vid = ImageSequenceClip(detected_frames, fps = orig_vid.fps)
new_vid.write_videofile('test_video/bear sits next to guy_detected.mp4')
  

  