# This script experiments with object character recognition (OCR).

# I use the Google Cloud Vision OCR API and
# an open source method.

################
# IMPORT MODULES
################
import os
import pandas as pd
import cv2
import numpy as np

import io                                        # Google API packages
from google.cloud import vision

from PIL import Image
import pytesseract
import argparse

####################################
# SET UP GOOGLE CLOUD VISION OCR API
####################################
# Instantiate a client.
# You need a Google Cloud account and service account key.
client = vision.ImageAnnotatorClient()

###########################################
# RUN GOOGLE CLOUD VISION OCR API ON IMAGES
###########################################

# Create results data frame.
results = pd.DataFrame(columns = ['file', 'method', 'text', 'x1', 'y1',
                                  'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])

# =============================================================================
#  # Loop through images.
os.chdir('..')
for file in os.listdir('Images/ocr_testing/')[0:1]:
#  
#      # Proceed if file is a JPEG.
#      if file.endswith('.jpg'):
#          
#          # Google API.
#  
#          # Load image into memory for Google API..
#           with io.open('Images/ocr_testing/' + file, 'rb') as google_image:
#               google_content = google_image.read()        
#   
#           # Modify loaded image.
#           google_image_mod = vision.types.Image(content = google_content)
#   
#           # Detect text in image.
#           google_response = client.text_detection(image = google_image_mod)
#   
#           # Get more digestible results.
#           google_response_mod = google_response.text_annotations
#   
#           # Loop through text results.
#           for google_entry in google_response_mod:
#   
#               # Append relevant stuff to 'results'.
#               results = results.append({'file': file,
#                                         'method': 'google',
#                                         'text': google_entry.description,
#                                         'x1': google_entry.bounding_poly.vertices[0].x,
#                                         'y1': google_entry.bounding_poly.vertices[0].y,
#                                         'x2': google_entry.bounding_poly.vertices[1].x,
#                                         'y2': google_entry.bounding_poly.vertices[1].y,
#                                         'x3': google_entry.bounding_poly.vertices[2].x,
#                                         'y3': google_entry.bounding_poly.vertices[2].y,
#                                         'x4': google_entry.bounding_poly.vertices[3].x,
#                                         'y4': google_entry.bounding_poly.vertices[3].y},
#                                        ignore_index = True)
# =============================================================================
             
        # 'pytesseract' method.
        
        # Load image in grayscale.
        im = cv2.imread(('Images/ocr_testing/' + str(file)), 0)
        
        # Blur image.
        #im = cv2.medianBlur(im, 1)
        
        # Simple thresholding.
        #_, im = cv2.threshold(im, 90, 255, cv2.THRESH_TOZERO)
        
        # Adaptive thresholding.
        #im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        #                           cv2.THRESH_BINARY, 21, 2)
        im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 91, 2)
        
        resize_im = cv2.resize(im, (round((im.shape[1] / 2)),
                     round(im.shape[0] / 2)))
        #cv2.imshow('', resize_im)
        #cv2.waitKey(0)
        
        #at = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        
         # Load image.
         #pyt_text = pytesseract.image_to_string(Image.open('Images/ocr_testing/' + file))
        pyt_text = pytesseract.image_to_string(im, 
                                               config = '--psm 11 -oem 2 -c tessedit_char_whitelist=0123456789')
        print(file)
        print(pyt_text)
        print('')
        print('')

####################################
# ANNOTATE IMAGE WITH TEXT AND BOXES
####################################
# This section draws the bounding boxes and text detected.

# =============================================================================
#  # Get image.
# image = cv2.imread('Images/ocr_testing/1975_Premium Cashier Cooler_6192578.jpg')
#  
#  # Get subset of 'results' corresponding to 'image'.
# im_results = results[results.file == '1975_Premium Cashier Cooler_6192578.jpg'].reset_index(drop = True)
#  
#  # Loop through text.
# for row in range(0, im_results.shape[0]):
#  
#      # Specify vertices and reshape for plotting.
#      vertices = np.array([[im_results.x1[row], im_results.y1[row]],
#                           [im_results.x2[row], im_results.y2[row]],
#                           [im_results.x3[row], im_results.y3[row]],
#                           [im_results.x4[row], im_results.y4[row]]], np.int32)
#      vertices = vertices.reshape((-1, 1, 2))
#  
#      # Add vertices and color of rectangle to image.
#      image = cv2.polylines(image, [vertices], True, (20, 229, 65))
#  
#  # Show image.
# image = cv2.resize(image, (round((image.shape[1] / 2)),
#                     round(image.shape[0] / 2)))
# cv2.imshow('', image)
# cv2.waitKey(0)
# =============================================================================




        
