#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:13:39 2019

@author: hankui
"""

#%% callback function
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True



#%% instantiate callback
callbacks = myCallback()


#%%
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()    


#%% design the model 
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (220,220,3)),
        tf.keras.layers.Dense(512, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)])

#%%
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')


#%%
model.fit(training_images, training_labels, epochs = 5, callbacks=[callbacks])
    


#%% Week 4 materials
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)


#%% point to the data directory, instead of the subdirectory
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(300,300),
        batch_size=128,
        class_mode='binary')


validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size=(300,300),
        batch_size=32,
        class_mode='binary')  


#%%
history = model.fit_generator(
        train_generator,
        steps_per_epoch = 8,
        epochs = 15,
        validation_data = validation_generator,
        validataion_steps = 8,
        verbose = 2)

































