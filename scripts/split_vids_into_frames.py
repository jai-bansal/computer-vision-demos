# This script splits videos into frames and saves those frames.
# The video frames will be used to qualitatively test a trained object 
# detection model.

################
# IMPORT MODULES
################
import cv2                          # OpenCV module
import os                           # used for file manipulation

################
# SET PARAMETERS
################

# Set path to videos.
# This is set as if the working directory is the "computer_vision_demos" 
# folder.
vids_path = 'test_video'

# Set path to where video frames are saved.
image_path = 'test_video_frames'

##########################
# SPLIT VIDEOS INTO FRAMES
##########################
# This section splits videos into frames and saves those frames.

# Loop through videos.
for file in os.listdir(vids_path):

    # Only proceed for '.mp4' files.
    if file.endswith('.mp4'):
        print('Splitting: ', file)

        # Import video.
        vid = cv2.VideoCapture(vids_path + '/' + file)

        # Get first frame of 'vid'.
        valid_input, frame = vid.read()

        # Save valid frames.
        while valid_input:           
           
            # Save frame.
            cv2.imwrite((image_path + '/' + str(file)[:-4] + '_' + str(vid.get(cv2.CAP_PROP_POS_FRAMES)) + '.jpg'),
                        frame)
    
            # Get next frame.
            valid_input, frame = vid.read()
