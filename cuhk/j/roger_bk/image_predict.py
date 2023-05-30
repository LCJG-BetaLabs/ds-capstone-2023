# Databricks notebook source
! ls /dbfs/team_j

# COMMAND ----------

! ls /dbfs/team_j/image_dataset/

# COMMAND ----------

# ! df -h /dbfs/*

# COMMAND ----------

from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
# from keras.applications.MobileNetV2 import preprocess_input
from tensorflow.keras.applications.MobileNetV2 import preprocess_input

# COMMAND ----------



# COMMAND ----------

def batch_predict(model_path, img_path):
    IMG_SIZE = (128, 128)

    # Load the model from the checkpoint
    #model_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426'
    model = load_model(model_path)

    # Load an image to predict
    #img_path = '/dbfs/team_j/image_dataset/9781503752139.jpg'
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    # Make a prediction
    pred = model.predict(x)
    #print(pred[0][0])
    return

# COMMAND ----------



# COMMAND ----------

# class load_model_and_predict:

class ModelPredictor:

    IMG_SIZE = (128, 128)

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model(self.model_path)


    def predict(self, img_path):
        img = load_img(img_path, target_size=self.IMG_SIZE)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)
        return pred[0][0]
        #print("My value is:", self.value)

# COMMAND ----------

m1 = ModelPredictor(model_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426')

# COMMAND ----------

m1.predict(img_path = '/dbfs/team_j/image_dataset/9781503752139.jpg')

# COMMAND ----------

import os

dir_path = '/dbfs/team_j/image_dataset/'

for item in os.listdir(dir_path):
    print(
        item,
        m1.predict(img_path = f"{dir_path}{item}") 
    )

# COMMAND ----------

# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load the model architecture
# model = tf.keras.models.load_model('/dbfs/team_j/model_checkpoint_roger_binary_class_20230426_2_with_oversample')


# COMMAND ----------

# # Load the image
# image_path = '/dbfs/team_j/image_dataset/9781503752139.jpg'
# image = Image.open(image_path).resize((128, 128))

# # Convert the image to a numpy array
# image_array = np.array(image)

# # Scale the pixel values to be between 0 and 1
# image_array = image_array / 255.0

# # Add an extra dimension to the array to make it compatible with the model
# image_array = np.expand_dims(image_array, axis=0)

# # Use the model to predict the class of the image
# predictions = model.predict(image_array)

# # Print the predictions
# print(predictions)

# COMMAND ----------

# Load the model architecture
model = tf.keras.models.load_model('/dbfs/team_j/model_checkpoint_roger_binary_class_20230426_2_with_oversample')



# COMMAND ----------

# Load the image
image_path = '/dbfs/team_j/image_dataset/9781503752139.jpg'
image = Image.open(image_path).resize((128, 128))

# Convert the image to a numpy array
image_array = np.array(image)

# Convert the image to a float32 array and scale the pixel values to be between 0 and 1
image_array = image_array.astype(np.float32) / 255.0

# Add an extra dimension to the array to make it compatible with the model
image_array = np.expand_dims(image_array, axis=0)

# Use the model to predict the class of the image
predictions = model.predict(image_array)

# Print the predictions
print(predictions)


# COMMAND ----------

# class load_model_and_predict:

class ModelPredictor:

    IMG_SIZE = (128, 128)

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model(self.model_path)

    def predict(self, img_path):
        #img = load_img(img_path, target_size=self.IMG_SIZE)
        img = Image.open(img_path).resize((128, 128))
        x = np.array(img)
        # x = img_to_array(img)
        x = x.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)
        return pred[0][0]
        #print("My value is:", self.value)

# COMMAND ----------

m1 = ModelPredictor(model_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426_2_with_oversample')

# COMMAND ----------

m1.predict(img_path = '/dbfs/team_j/image_dataset/9781503752139.jpg')

# COMMAND ----------

import os

dir_path = '/dbfs/team_j/image_dataset/'

for item in os.listdir(dir_path):
    print(
        item,
        m1.predict(img_path = f"{dir_path}{item}") 
    )

# COMMAND ----------


m2 = ModelPredictor(model_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426_2_with_oversample_vgg16')


# COMMAND ----------

import os

dir_path = '/dbfs/team_j/image_dataset/'

for item in os.listdir(dir_path):
    print(
        item,
        m2.predict(img_path = f"{dir_path}{item}") 
    )

# COMMAND ----------

m3 = ModelPredictor(model_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426')

# COMMAND ----------

import os

dir_path = '/dbfs/team_j/image_dataset/'

for item in os.listdir(dir_path):
    print(
        item,
        m3.predict(img_path = f"{dir_path}{item}") 
    )

# COMMAND ----------



# COMMAND ----------

