# This script does optical character recognition (OCR) on some sample images.
# This script is just to get OCR working, 
# so there's minimal image pre-processing.

# IMPORT LIBRARIES --------------------------------------------------------
library(tesseract)
library(magick)
library(here)

# CONDUCT OCR ON SAMPLE IMAGES --------------------------------------------
# This script loops through images and returns OCR results.

  # Set working directory.
  setwd(here('sample_images'))

  # Loop through images and conduct OCR.
  for file in list.files():
  
    ocr_result = image_read(file) %>%                      # Read in file
      
                 
  
    
    


text = image_read('955_Premium Cashier Cooler_6173676.jpg') %>% 
  
  image_resize('2000') %>% 
  
  image_convert(colorspace = 'gray') %>% 
  
  image_trim() %>% 
  
  image_ocr()

cat(text)
  
