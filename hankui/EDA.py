# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% read in images in 'sunflower' folder, approach 1
from PIL import Image
jpgfile = Image.open("6953297_8576bf4ea3.jpg") 
print(jpgfile.bits, jpgfile.size, jpgfile.format)


#%% read in one image, approach 2
import numpy as np
import cv2
import matplotlib.pyplot as plt


#%% Load an color image in grayscale
img = cv2.imread('sunflower/6953297_8576bf4ea3.jpg', 0)


#%% display the image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Normalize, rescale entries to lie in [0,1]
gray_img = img.astype("float32")/255


#%% define the filter
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])
print('Filter shape: ', filter_vals.shape)


#%% Define four different filters, all of which are linear combinations of the `filter_vals` defined above
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])


#%% Print out the values of filter 1 as an example
print('Filter 1: \n', filter_1)


#%% save a couple objects into a pickle file 
obj = [filter_1, filter_2]
import pickle

f = open('store.pckl', 'wb')
pickle.dump(obj, f)
f.close()


#%% read in the saved objects from the pickle file
import pickle
f = open('store.pckl', 'rb')
obj = pickle.load(f)
f.close()


#%% do the same thing as above, but now use shelve
import shelve

#T='Hiya'
#val=[1,2,3]

filename='shelve.out'
my_shelf = shelve.open(filename,'n') # 'n' for new


for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()


#%%
import shelve
filename='shelve.dat'

my_shelf = shelve.open(filename)
for key in my_shelf:
    my_shelf[key] = globals()[key]
my_shelf.close()


#%%
import shelve 
r = shelve.open("shelve.dat") 
for key in r:
   print(r[key]) 
r.close()















