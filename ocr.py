# This script does optical character recognition (OCR) on some sample images.
# This script is just to get OCR working, 
# so there's minimal image pre-processing.

################
# IMPORT MODULES
################
import os
import pandas as pd
import cv2
import numpy as np

from PIL import Image
import pytesseract
import argparse

##############################
# CONDUCT OCR ON SAMPLE IMAGES
##############################
# This script loops through images, runs OCR, and displays images
# with bounding boxes.

# Loop through images.
for file in os.listdir('sample_images')[0:1]:

              # Run OCR on image using Pytesseract.
              pyt_text = pytesseract.image_to_string(Image.open('sample_images/' + file))
              pyt_boxes = pytesseract.image_to_boxes(Image.open('sample_images/' + file))

              # Print text.
              print(pyt_text)
              print('')

              # Get image.
              image = cv2.imread('sample_images/' + file)

              # draw the bounding boxes on the image
              for b in pyt_boxes.splitlines():

                  # Split 'b' by space.    
                  b = b.split(' ')

                  # Add bounding boxes to image.
                  image = cv2.rectangle(image,
                                        (int(b[1]), image.shape[0] - int(b[2])),
                                        (int(b[3]), image.shape[0] - int(b[4])),
                                        (0, 255, 0), 2)

              cv2.imshow('', image)
              cv2.waitKey(0)





        
