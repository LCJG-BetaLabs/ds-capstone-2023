# Databricks notebook source
pip install opencv-python

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D, BatchNormalization
tf.__version__

pd.set_option('display.max_colwidth', None)

# COMMAND ----------

def img_path(atg_code):
    if atg_code[0:5] == 'check':
        return f"/dbfs/FileStore/tables/image/checks/{atg_code}.jpg"
    if atg_code[0:8] == 'diamonds':
        return f"/dbfs/FileStore/tables/image/diamonds/{atg_code}.jpg"
    if atg_code[0:6] == 'strips':
        return f"/dbfs/FileStore/tables/image/strips/{atg_code}.jpg"
    else :
        return f"/dbfs/image/{atg_code}_in_xl.jpg"

def load_image_original(atg_code, resized_fac = 0.5):
    img     = cv2.imread(img_path(atg_code))
    #h, _, _ = img.shape
    #img   = img[0:int(h*2/3), :, :] # preserve upper half of the image only
    h, w, _ = img.shape
    resized = cv2.resize(img, (int(w*resized_fac), int(h*resized_fac)), interpolation = cv2.INTER_AREA)
    return resized

def load_image(atg_code, resized_fac = 0.5):
    if atg_code[0:5] != 'check' and atg_code[0:8] != 'diamonds':
        img     = cv2.imread(img_path(atg_code))
        h, _, _ = img.shape
        img   = img[0:int(h*2/3), :, :] # preserve upper half of the image only
        h, w, _ = img.shape
        resized = cv2.resize(img, (int(w*resized_fac), int(h*resized_fac)), interpolation = cv2.INTER_AREA)
        return resized
    else:
        img     = cv2.imread(img_path(atg_code))
        h, w, _ = img.shape
        resized = cv2.resize(img, (int(w*resized_fac), int(h*resized_fac)), interpolation = cv2.INTER_AREA)
        return resized

def plot_figures(figures, nrows = 1, ncols=1,figsize=(8, 8)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=figsize)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional

# COMMAND ----------

# show image path
import pyspark.dbutils
image_path_checks = dbutils.fs.ls("file:/dbfs/FileStore/tables/image/checks/")
image_path_diamonds = dbutils.fs.ls("file:/dbfs/FileStore/tables/image/diamonds/")
image_path_strips = dbutils.fs.ls("file:/dbfs/FileStore/tables/image/strips/")

# COMMAND ----------

# DBTITLE 1,Use Pre-Trained Model to Recommendation
# show image and how we cut the image in half
img_array = load_image("BWJ579")
img_array2 = load_image_original("BWJ579")

plt.figure()
f, axarr = plt.subplots(1,2) 
axarr[0].imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
axarr[1].imshow(cv2.cvtColor(img_array2, cv2.COLOR_BGR2RGB))

# COMMAND ----------

def get_embedding(model, img_name):
    # Reshape
    img = image.load_img(img_path(img_name), target_size=(img_width, img_height))
    # img to Array
    x   = image.img_to_array(img)
    # Expand Dim (1, w, h)
    x   = np.expand_dims(x, axis=0)
    # Pre process Input
    x   = preprocess_input(x)
    return model.predict(x).reshape(-1)

# COMMAND ----------

# DBTITLE 1,Obtain embedding with ResNet50 (with additional dataset)
from tensorflow.keras.applications import ResNet50

# Input Shape
img_width, img_height, color = load_image("BUE682").shape

# Pre-Trained Model
base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape = (img_width, img_height, color))
base_model.trainable = False

# Add Layer Embedding
model = keras.Sequential([
    base_model,
    GlobalMaxPooling2D(),
    BatchNormalization(), # add for the combined model only
    #layers.Dense(1024, activation="relu")
])

model.summary()

# COMMAND ----------

import pandas as pd
import os
df = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df.head()

# COMMAND ----------

df_old_data = df[['atg_code', 'class']]

df_new_data = pd.DataFrame([file_info.name.split('.')[0] for file_info in image_path_checks])
df_new_data['class'] = 8
df_new_data.columns = ['atg_code', 'class']

df_new_data2 = pd.DataFrame([file_info.name.split('.')[0] for file_info in image_path_diamonds])
df_new_data2['class'] = 9
df_new_data2.columns = ['atg_code', 'class']

# df_new_data2 = pd.DataFrame([file_info.name.split('.')[0] for file_info in image_path_strips])
# df_new_data2['class'] = 10
# df_new_data2.columns = ['atg_code', 'class']

df_sample = pd.concat([df_old_data, df_new_data, df_new_data2])
df_sample.columns = [['atg_code', 'Y']]
df_sample = df_sample.reset_index(drop=True)
df_sample


# COMMAND ----------

df_embs = pd.DataFrame()
print(df_sample.shape[0])
for idx, items in df_sample['atg_code'].iterrows():
    nparray = get_embedding(model, items.atg_code)
    new_embs = pd.DataFrame(nparray.reshape(-1, len(nparray)))
    df_embs = df_embs.append([new_embs])
    print(idx)

print(df_embs.shape)
df_embs.to_csv('/dbfs/image_embedding_data_new_50percent_upper.csv')

df_embs = df_embs.reset_index(drop=True)
df_embs['atg_code'] = df_sample['atg_code']
df_embs['Y'] = df_sample['Y']
df_embs.to_csv('/dbfs/image_embedding_data_new_50percent_upper.csv', index=False)
df_embs.groupby('Y').size()

# COMMAND ----------

# DBTITLE 1,Obtain embedding with ResNet101
from tensorflow.keras.applications import ResNet101

# Input Shape
img_width, img_height, color = load_image("BUE682").shape

# Pre-Trained Model
base_model = ResNet101(weights='imagenet', 
                      include_top=False, 
                      input_shape = (img_width, img_height, color))
base_model.trainable = False

# Add Layer Embedding
model = keras.Sequential([
    base_model,
    GlobalMaxPooling2D(),
    BatchNormalization(), # add for the combined model only
    #layers.Dense(1024, activation="relu")
])

model.summary()

# COMMAND ----------

df_sample      = df
map_embeddings = df_sample['atg_code'].apply(lambda img: get_embedding(model, img))
df_embs        = map_embeddings.apply(pd.Series)

print(df_embs.shape)
df_embs['atg_code'] = df_sample['atg_code']
df_embs['Y'] = df_sample['class']
df_embs.to_csv('/dbfs/image_embedding_data_restnet101.csv', index=False)
df_embs.head()

# COMMAND ----------

# DBTITLE 1,Obtain embedding with ResNet152
from keras.applications.resnet_v2 import ResNet152V2

# Input Shape
img_width, img_height, color = load_image("BUE682").shape

# Pre-Trained Model
base_model = ResNet152V2(weights='imagenet', 
                      include_top=False, 
                      input_shape = (img_width, img_height, color))
base_model.trainable = False

# Add Layer Embedding
model_152 = keras.Sequential([
    base_model,
    GlobalMaxPooling2D(),
    BatchNormalization(), # add for the combined model only
    #layers.Dense(1024, activation="relu")
])

model_152.summary()

# COMMAND ----------

df_sample      = df
map_embeddings = df_sample['atg_code'].apply(lambda img: get_embedding(model_152, img))
df_embs        = map_embeddings.apply(pd.Series)

print(df_embs.shape)
df_embs['atg_code'] = df_sample['atg_code']
df_embs['Y'] = df_sample['class']
df_embs.to_csv('/dbfs/image_embedding_data_restnet152.csv', index=False)
df_embs.head()