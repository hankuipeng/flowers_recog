# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% read in images, approach 1
from PIL import Image
jpgfile = Image.open("6953297_8576bf4ea3.jpg") 
print(jpgfile.bits, jpgfile.size, jpgfile.format)


#%% read in one image, approach 2
import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('6953297_8576bf4ea3.jpg', 0)


#%% display the image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


