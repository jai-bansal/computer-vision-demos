# This script does optical character recognition (OCR) on some sample images.
# This script is just to get OCR working, 
# so there's minimal image pre-processing.

# For some reason, the output colors for the 2nd image are totally wrong...
# The returned text is correct though!

################
# IMPORT MODULES
################
import os
import cv2

from PIL import Image
import pytesseract

##############################
# CONDUCT OCR ON SAMPLE IMAGES
##############################
# This script loops through images, runs OCR, and displays images
# with bounding boxes.

# Loop through images.
for file in os.listdir('sample_images'):

              # Run OCR on image and get text.
              pyt_text = pytesseract.image_to_string(Image.open('sample_images/' + file))

              # Print text.
              print(pyt_text)
              print('')

              # Run OCR on image and get bounding box info.
              pyt_boxes = pytesseract.image_to_boxes(Image.open('sample_images/' + file))

              # Restructure bounding box info.
              pyt_boxes = pyt_boxes.splitlines()

              # Get image.
              image = cv2.imread('sample_images/' + file)

              # Draw bounding boxes on image.
              for coords in pyt_boxes:

                  # Each 'coords' is a string of the coordinates separated by spaces.
                  # Split 'coords' by space.    
                  coords = coords.split(' ')

                  # Add bounding boxes to image.
                  # 'coords' has 6 elements. The first and last are NOT actual
                  # bounding box coordinates.
                  image = cv2.rectangle(image,
                                        (int(coords[1]),
                                         image.shape[0] - int(coords[4])),
                                        (int(coords[3]),
                                         image.shape[0] - int(coords[2])),
                                        (0, 255, 0), 2)

              # Display image.
              cv2.imshow('', image)
              cv2.waitKey(0)

              # Clean up.
              del(pyt_text, pyt_boxes, image, coords)





        
