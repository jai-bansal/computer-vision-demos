# This script does optical character recognition (OCR) on some sample images.
# This script is just to get OCR working, 
# so there's minimal image pre-processing.

# IMPORT LIBRARIES --------------------------------------------------------
library(tesseract)
library(magick)
library(here)
library(tibble)
library(dplyr)
library(tidyr)

# CONDUCT OCR ON SAMPLE IMAGES --------------------------------------------
# This script loops through images and returns OCR results.

  # Set working directory.
  setwd(here('sample_images'))

  # Create result data frames.
  bbox_results = tibble()

  # Loop through images and conduct OCR.
  for (file in list.files())
    
  {
    
    # Keep track of progress.
    print(file)
    
    # Process image.
    image_data = image_read(file) %>%                      # Read in file
      
                 image_convert(colorspace = 'gray')       # Convert image to grayscale
    
    # Conduct OCR and append results to 'text_results'.
    text = image_data %>% image_ocr()
    
    # Display OCR result.
    cat(text)
    
    # Get bounding box results.
    bbox = image_data %>% 
      
           ocr_data() %>%                                  # Get confidence and bounding box data.
      
           mutate(file = file)                             # Add file column.
    
    # Append 'bbox' to 'bbox_results'.
    bbox_results = rbind(bbox_results, bbox)

  }
  
# SHOW BOUNDING BOXES FOR EACH IMAGE --------------------------------------
# This section shows the bounding boxes for the detected text.
  
  # Create one column for each bounding box coordinate.
  bbox_results = bbox_results %>% 
    
                 separate(col = bbox,                               # Separate 'bbox' column into coordinates.
                          into = c('x1', 'y1', 'x2', 'y2'), 
                          sep = ',')
  
  # Image 1.
  
    # Read in image.
    i1 = image_read(list.files()[1])
  
    # Draw image.
    i1 = image_draw(i1)

    # Create subset for just 'i1' words.
    i1_boxes = filter(bbox_results, file == 'text1.png')
        
    # Add rectangles and print image.
    rect(xleft = i1_boxes$x1, 
         xright = i1_boxes$x2, 
         ytop = i1_boxes$y1, 
         ybottom = i1_boxes$y2, 
         border = 'green')
    
  # Image 2.
    
    # Read in image.
    i2 = image_read(list.files()[2])
    
    # Draw image.
    i2 = image_draw(i2)
    
    # Create subset for just 'i2' words.
    i2_boxes = filter(bbox_results, file == 'text2.png')
    
    # Add rectangles and print image.
    rect(xleft = i2_boxes$x1, 
         xright = i2_boxes$x2, 
         ytop = i2_boxes$y1, 
         ybottom = i2_boxes$y2, 
         border = 'green')
    
  # Image 3.
    
    # Read in image.
    i3 = image_read(list.files()[3])
    
    # Draw image.
    i3 = image_draw(i3)
    
    # Create subset for just 'i3' words.
    i3_boxes = filter(bbox_results, file == 'text3.png')
    
    # Add rectangles and print image.
    rect(xleft = i3_boxes$x1, 
         xright = i3_boxes$x2, 
         ytop = i3_boxes$y1, 
         ybottom = i3_boxes$y2, 
         border = 'green')
