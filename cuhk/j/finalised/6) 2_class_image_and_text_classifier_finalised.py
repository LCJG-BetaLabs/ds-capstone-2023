# Databricks notebook source
# MAGIC %md # this notebook shows the steps to train a binary ANN keras classifier that:
# MAGIC   - takes in **image** data and `description` of a book as predictors
# MAGIC   - predicts a binary variable `is_high_sales_volume` (flagged as 1 or 0)
# MAGIC     - `is_high_sales_volume` is derived from a cutoff point that determines whether a book falls into high or low sales, depending on 2021~2022 dataset from `RECOMMENDATION_*.csv` dataset
# MAGIC       - 65th percentile of `QUANTITY` distribution (aggregated view by book) serves as the cut off point
# MAGIC         - `1` denotes as high sales
# MAGIC         - `0` denotes as low sales

# COMMAND ----------

# MAGIC %md # install packages

# COMMAND ----------

!pip install tensorflow --upgrade

# COMMAND ----------

!pip install tensorflow_hub

# COMMAND ----------

!pip install bert

# COMMAND ----------

!pip install opencv-python

# COMMAND ----------

!pip install --upgrade bert

# COMMAND ----------

!pip install tokenization

# COMMAND ----------

!pip install bert-tensorflow

# COMMAND ----------

# MAGIC %md # import packages

# COMMAND ----------

import tensorflow as tf

# COMMAND ----------

# Import all necessary libraries
try:
    %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import applications
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.regularizers import l2, l1

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import bert

import numpy as np 
import pandas as pd
import re
import glob
import os
import cv2
import sys
import pickle

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# import plotly.express as px
import plotly.graph_objects as go
import os


# from tensorflow.keras.applications import InceptionV3
from transformers import BertTokenizer


# COMMAND ----------

# MAGIC %md # Load the json files that stores ISBN and google book api fetched result:
# MAGIC   - book `description` is stored there

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm_20230421_2")
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_4.json").createOrReplaceTempView("isbn_google_reomm_20230421_4")

# COMMAND ----------

# MAGIC %md # prepare the dataset that has column ISBN and description

# COMMAND ----------

isbn_desc = spark.sql("""
    with exploded_2 as (
    select 
        explode(items)
    from isbn_google_reomm_20230421_2
    ),
    volumeinfo_2 AS (
    select 
    col.volumeInfo.*
    FROM
        exploded_2
    ),
    exploded_4 as (
    select 
        explode(items)
    from isbn_google_reomm_20230421_4
    ),
    volumeinfo_4 AS (
    select 
    col.volumeInfo.*
    FROM
        exploded_2
    ),
    unioned AS (
    select * from volumeinfo_2
    union all 
    select * from volumeinfo_4
    ),
    isbn_image_link_pair AS (
    SELECT
        distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
        imageLinks.thumbnail as thumbnail,
        description
    FROM
        unioned
    )
    select 
    isbn,
    description
    from 
    isbn_image_link_pair
    where 
    thumbnail is not null and description is not null 
""").toPandas()

# COMMAND ----------

# MAGIC %md # load `RECOMMENDATION_*` csv file for label variable `is_high_sales_volume`

# COMMAND ----------

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 
df_reomm_p = df_reomm.toPandas() # padnas 


# COMMAND ----------

# MAGIC %md # a function that do the dataframe pre-processing on `RECOMMENDATION_*` csv file
# MAGIC   - drop mal-formed row
# MAGIC   - convert needed columns into an appropriate data type:
# MAGIC     - `PRICE`, `QUANTITY` and `AMOUNT`
# MAGIC   - add columns `year` , `month` and `day` from `TRANDATE`

# COMMAND ----------

def clean_recomm_df(df: pd.DataFrame) -> pd.DataFrame:

    # df_reomm_p_2 = df_reomm_p[df_reomm_p["HASHED_INVOICE_ID"].apply(lambda s: s.startswith("0x"))]
    # df_reomm_p_2 = df_reomm_p_2[~df_reomm_p_2['QUANTITY'].isnull()]
    # df_reomm_p_2 = df_reomm_p_2.drop("ISBN13", axis=1)

    # df_reomm_p_2["PRICE"] = df_reomm_p_2["PRICE"].astype(float)
    # df_reomm_p_2["QUANTITY"] = df_reomm_p_2["QUANTITY"].astype(int)
    # df_reomm_p_2["AMOUNT"] = df_reomm_p_2["AMOUNT"].astype(float)

    # df_reomm_p_2["year"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[0])
    # df_reomm_p_2["month"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[1])
    # df_reomm_p_2["day"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[2])


    ########################################################################################################################################################
    #   ref: https://adb-5911062106551859.19.azuredatabricks.net/?o=5911062106551859#notebook/3108408038812593/command/751034215087416                     #
    ########################################################################################################################################################

    df_2 = df[df["HASHED_INVOICE_ID"].apply(lambda s: s.startswith("0x"))]
    df_2 = df_2[~df_2['QUANTITY'].isnull()]
    #df_2 = df_2.drop("ISBN13", axis=1)

    df_2['ISBN13'] = df_2['ISBN13'].apply(lambda s:s.rstrip())

    df_2["PRICE"] = df_2["PRICE"].astype(float)
    df_2["QUANTITY"] = df_2["QUANTITY"].astype(int)
    df_2["AMOUNT"] = df_2["AMOUNT"].astype(float)

    df_2["year"] = df_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[0])
    df_2["month"] = df_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[1])
    df_2["day"] = df_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[2])

    return df_2

# COMMAND ----------

# MAGIC %md # further data pre-processing on `RECOMMENDATION_*` csv file and create another dataframe that stores the aggregated view by product
# MAGIC   - drop rows whose title is `Group Cash Coupon - $100`
# MAGIC   - group by the continuous variable throughout the time line by ISBN13

# COMMAND ----------

df_reomm_p_cleaned = clean_recomm_df(df = df_reomm_p)
df_reomm_p_cleaned_wo_coupon = df_reomm_p_cleaned[df_reomm_p_cleaned['TITLE'] != "Group Cash Coupon - $100"]  # exclude this item
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon.groupby("ISBN13").sum().reset_index()
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon_stat_by_isbn.sort_values(by='QUANTITY', ascending=False)

# COMMAND ----------

# MAGIC %md # obtain the 65th data point from `QUANTITY` distribution 

# COMMAND ----------

q_65 = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].quantile(0.65)
df_reomm_p_cleaned_wo_coupon_stat_by_isbn['is_high_sales_volume'] = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].apply(lambda x: 1 if x >= q_65 else 0)

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon_stat_by_isbn

# COMMAND ----------

# MAGIC %md # prepare the dataset that joins 1 dataframe that has `ISBN` and `description` and 1 dataframe that has aggregated `PRICE`, `QUANTITY` and `AMOUNT`

# COMMAND ----------

df_text_image_dataset = pd.merge(isbn_desc, df_reomm_p_cleaned_wo_coupon_stat_by_isbn, left_on='isbn', right_on='ISBN13', how='inner')

# COMMAND ----------

# MAGIC %md # data transformation on column `isbn` to add `.jpg` suffix for later-on image data querying during model training

# COMMAND ----------

df_text_image_dataset['isbn'] = df_text_image_dataset['isbn'].apply(lambda x: x + ".jpg")
df_text_image_dataset = df_text_image_dataset[['isbn', 'description', 'is_high_sales_volume']]

# COMMAND ----------

# MAGIC %md # `bert-base-uncased` is used in the tokenizer for pre-processing text `description` variable 

# COMMAND ----------

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# COMMAND ----------

# MAGIC %md function to pre-process the image dataset before training

# COMMAND ----------

# Define function to load image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


# COMMAND ----------

# MAGIC %md # prepare `18000`-rows dataset with `description` and image for training

# COMMAND ----------

class_0_samples = df_text_image_dataset[df_text_image_dataset['is_high_sales_volume'] == 0].sample(n=9000, random_state=42)
class_1_samples = df_text_image_dataset[df_text_image_dataset['is_high_sales_volume'] == 1].sample(n=9000, random_state=42)
df = pd.concat([class_0_samples, class_1_samples])

df['isbn'] = df['isbn'].astype(str)
df['is_high_sales_volume'] = df['is_high_sales_volume'].astype(str)

# train dataset is used on below code. this code is not the true train_test_split !!!! 
train, test = train_test_split(df, test_size=0.001, random_state=42)

# COMMAND ----------


# Load images and preprocess text for training data
train_images = []
train_texts = []
train_labels = []
for index, row in train.iterrows():
    # Load image
    image_path = row['isbn']
    img = load_image('/dbfs/team_j/image_dataset/' +image_path)
    train_images.append(img)

    # Preprocess text
    text = row['description']
    
    encoded_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='tf')
    train_texts.append(encoded_text['input_ids'])

    # Get label
    label = row['is_high_sales_volume']
    train_labels.append(label)

# Convert lists to numpy arrays
train_images = np.array(train_images)
# train_texts = np.array(train_texts)
train_texts = np.squeeze(np.array(train_texts))
# train_labels = np.array(train_labels)
# train_labels = train_labels.astype('float32')

# COMMAND ----------

train_labels = np.array(train_labels).astype('float32')

# COMMAND ----------

# MAGIC %md # VGG16 pretrain layer is used 

# COMMAND ----------

from tensorflow.keras.applications import VGG16

# COMMAND ----------

# MAGIC %md # define a concatenated image + text model 
# MAGIC   - VGG16 as pretrained layeres

# COMMAND ----------

# # Define image model with VGG16

image_model = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
image_model.trainable = False
x = image_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x) 
x = layers.Dense(256, activation='relu')(x) 
x = layers.Dense(128, activation='relu')(x) 
image_model = tf.keras.Model(inputs=image_model.input, outputs=x)


# # Define text model with LSTM
# text_input = layers.Input(shape=(512,), dtype='int32')
# text_embedding = layers.Embedding(len(tokenizer.get_vocab()), 128)(text_input)
# text_lstm = layers.LSTM(128)(text_embedding)
# text_lstm = layers.Dense(256, activation='relu')(text_lstm) 

text_input = layers.Input(shape=(512,), dtype='int32')
text_embedding = layers.Embedding(len(tokenizer.get_vocab()), 128)(text_input)
text_conv1 = layers.Conv1D(128, 5, activation='relu')(text_embedding)
text_pool1 = layers.MaxPooling1D(5)(text_conv1)
text_conv2 = layers.Conv1D(128, 5, activation='relu')(text_pool1)
text_pool2 = layers.GlobalMaxPooling1D()(text_conv2)
text_dense = layers.Dense(256, activation='relu')(text_pool2)

# Concatenate image and text models
concatenated = layers.concatenate([image_model.output, text_dense])
dense1 = layers.Dense(512, activation='relu')(concatenated)
dense1 = layers.Dense(256, activation='relu')(dense1) 
output = layers.Dense(1, activation='sigmoid')(dense1)
model = tf.keras.models.Model(inputs=[image_model.input, text_input], outputs=output)


# COMMAND ----------

# DBTITLE 1,Running model training 
# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up checkpoint path
checkpoint_path = "/dbfs/team_j/text_image_combined_model_roger_20230427_vgg16_cnn_for_text_epoch5_trial_2"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# train
history = model.fit([train_images, train_texts], train_labels, validation_split=0.2, batch_size=32, epochs=5, callbacks=[cp_callback])

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

