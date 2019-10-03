#%%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


#%% check the number of pictures in each class 
import os

daisy_dir = os.path.join('flowers/daisy')
dandelion_dir = os.path.join('flowers/dandelion')
rose_dir = os.path.join('flowers/rose')
sunflower_dir = os.path.join('flowers/sunflower')
tulip_dir = os.path.join('flowers/tulip')

print('total training daisy images:', len(os.listdir(daisy_dir))) #769
print('total training dandelion images:', len(os.listdir(dandelion_dir))) #1055
print('total training rose images:', len(os.listdir(rose_dir))) #784
print('total training sunflower images:', len(os.listdir(sunflower_dir))) #736
print('total training tulip images:', len(os.listdir(tulip_dir))) #984


#%% visualize some pictures
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sunflower_files = os.listdir(sunflower_dir)

pic_index = 2

next_flower = [os.path.join(sunflower_dir, fname) 
                for fname in sunflower_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_flower):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()


#%% 
# data augmentation
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'flowers/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        #batch_size=128,
        class_mode='categorical') # 'categorical' for multi-class, and 'binary' for two-class


#%% construct the model
K = 5; # number of classes 
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(K, activation='softmax') # 5 for 5 classes
    ## tf.keras.layers.Dense(1, activation='sigmoid') 
])

model.summary()


#%%
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


#%%
history = model.fit_generator(
      train_generator,
      #steps_per_epoch=8,  
      epochs=100, # run for 100 epochs
      verbose=1)


#%% save model outputs 
# reference: https://machinelearningmastery.com/save-load-keras-deep-learning-models/  

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


#%% load json and create model
from keras.models import model_from_json
from keras.models import load_model

#with open('model.json','r') as f:
#    json = f.read()
#    loaded_model = model_from_json(json)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


