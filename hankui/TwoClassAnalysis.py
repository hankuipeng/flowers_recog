# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% import necessary packages
import os
import tensorflow as tf


#%% read in the training data 
# Directory with our training rose pictures
train_rose_dir = os.path.join('/home/hankui/Dropbox/Ongoing/flowers_recognition_hankui/flowers/rose')

# Directory with our training sunflower pictures
train_sunflower_dir = os.path.join('/home/hankui/Dropbox/Ongoing/flowers_recognition_hankui/flowers/sunflower')


#%% check how the file names look like
train_rose_names = os.listdir(train_rose_dir)
train_sunflower_names = os.listdir(train_sunflower_dir)

print(train_rose_names[:10])
print(train_sunflower_names[:10])


#%% check the number of pictures within each folder
print('total training horse images:', len(os.listdir(train_rose_dir)))
print('total training human images:', len(os.listdir(train_sunflower_dir)))


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
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
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

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


