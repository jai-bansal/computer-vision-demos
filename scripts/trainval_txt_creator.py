# This script creates 'trainval.txt'.
# The file is put into the 'annotations' folder. 
# This is apparently important for the whole object detection model to work.

################
# IMPORT MODULES
################
import os

################
# SET PARAMETERS
################
# This section provides the names of various folders to allow the script to 
# run in a more automated way.

# The folders below should be on the same directory level.

# Specify folder containing annotations.
annotation_folder = 'annotations'

# Specify folder containing images.
image_folder = 'images'

###########################
# CREATE RELEVANT TEXT FILE
###########################

# Create text file.
f = open(annotation_folder + '/' + 'trainval.txt', 'w')

for file in os.listdir(image_folder):
    f.write(file.replace('.jpg', '') + ' 1')
    f.write('\n')

# Close text file.
f.close()