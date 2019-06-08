# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% load the necessary packages 
import os
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image
import numpy as np
import cv2


#%% read in the training data 
# Directory with our training rose pictures
train_rose_dir = os.path.join('/home/hankui/Dropbox/Ongoing/flowers_recognition_hankui/flowers/rose')

# Directory with our training sunflower pictures
train_sunflower_dir = os.path.join('/home/hankui/Dropbox/Ongoing/flowers_recognition_hankui/flowers/sunflower')


#%% function to read in all images in a fodler 
# source: https://www.quora.com/How-can-I-read-a-data-set-of-images-in-a-PNG-format-in-Python-code
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        #img = Image.open(os.path.join(folder,filename)) # read in the image
        img = image.load_img(os.path.join(folder,filename), target_size=(300,300))
        img_arr = image.img_to_array(img) # convert the image to numpy array 
        img_arr = np.expand_dims(img_arr, axis = 0)
        #images = np.vstack([img_arr])
        images.append(img_arr)
    return images


#%% read in the data 
roses = load_images_from_folder(train_rose_dir)        
sunflowers = load_images_from_folder(train_sunflower_dir)       
train_data = roses+sunflowers


#%% re-size all the data to have the same dimension
# source: https://towardsdatascience.com/image-pre-processing-c1aec0be3edf
# setting dim of the resize
height = 300
width = 300
dim = (width, height)
res_img = []

for i in range(len(train_data)):
        res = cv2.resize(train_data[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

train_data = res_img


#%% create the labels 
lab1 = np.ones(len(roses), dtype = int) 
lab2 = np.ones(len(sunflowers), dtype = int)*2
train_labels = np.concatenate((lab1, lab2), axis=0)


#%% check how the file names look like
train_rose_names = os.listdir(train_rose_dir)
train_sunflower_names = os.listdir(train_sunflower_dir)

print(train_rose_names[:10])
print(train_sunflower_names[:10])


#%% check the number of pictures within each folder
print('total training rose images:', len(os.listdir(train_rose_dir)))
print('total training sunflower images:', len(os.listdir(train_sunflower_dir)))


#%%
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0


#%%
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_rose_pix = [os.path.join(train_rose_dir, fname) 
                for fname in train_rose_names[pic_index-8:pic_index]]
next_sunflower_pix = [os.path.join(train_sunflower_dir, fname) 
                for fname in train_sunflower_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_rose_pix+next_sunflower_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


#%% 
img = mpimg.imread(train_rose_names[0])
len(img)
print(type(img))


#%%
for names in enumerate(train_rose_names+train_sunflower_names):
    
    # read the image in as ndarray
    dat_i = mpimg.imread(names[i])



#%% convert an image to grayscale
from PIL import Image 
image = Image.open(train_rose_names[100]).convert('LA') # convert image to grayscale
image
#plt.imshow(image)
print(type(image))


#%%
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])


#%%
model.summary()


#%%
from tensorflow.python.keras.optimizers import RMSprop

model.compile(loss = 'binary_crossentropy',
              optimizer = tf.train.AdamOptimizer(), # other option: RMSprop(lr=0.001)
              metrics=['acc'])

# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')


#%% 
model.fit(train_data, train_labels, epochs = 5)


#%%
# Preprocessing
def processing(img):

    # loading image
    N = len(img)
    #img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:N]]
    print('Original size',img[0].shape)
    
    # setting dim of the resize
    height = 220
    width = 220
    dim = (width, height)
    res_img = []
    
    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

    # Checcking the size
    print("RESIZED", res_img[1].shape)
    
    # Visualizing one of the images in the array
    original = res_img[1]
    display_one(original)


#%% re-size all the data to have the same dimension
# setting dim of the resize
height = 220
width = 220
dim = (width, height)
res_img = []

for i in range(len(train_data)):
        res = cv2.resize(train_data[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

train_data = res_img


#%%
np.shape(train_data[1])
