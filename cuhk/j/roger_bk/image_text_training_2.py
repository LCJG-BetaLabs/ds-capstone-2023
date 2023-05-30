# Databricks notebook source
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

# import bert_tokenizer as tokenization

# tokenization.tokenizer.FullTokenizer

# COMMAND ----------

# !pip install --upgrade protobuf

# COMMAND ----------

!pip uninstall numpy

# COMMAND ----------

!pip uninstall tensorflow

# COMMAND ----------

!pip install --ignore-installed --upgrade tensorflow==1.9

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

# COMMAND ----------

# MAGIC %md # dataset preparation

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm_20230421_2")
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_4.json").createOrReplaceTempView("isbn_google_reomm_20230421_4")

# COMMAND ----------



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

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 
df_reomm_p = df_reomm.toPandas() # padnas 


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

df_reomm_p_cleaned = clean_recomm_df(df = df_reomm_p)
df_reomm_p_cleaned_wo_coupon = df_reomm_p_cleaned[df_reomm_p_cleaned['TITLE'] != "Group Cash Coupon - $100"]  # exclude this item
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon.groupby("ISBN13").sum().reset_index()
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon_stat_by_isbn.sort_values(by='QUANTITY', ascending=False)

# COMMAND ----------

q_65 = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].quantile(0.65)
df_reomm_p_cleaned_wo_coupon_stat_by_isbn['is_high_sales_volume'] = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].apply(lambda x: 1 if x >= q_65 else 0)

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon_stat_by_isbn

# COMMAND ----------

df_text_image_dataset = pd.merge(isbn_desc, df_reomm_p_cleaned_wo_coupon_stat_by_isbn, left_on='isbn', right_on='ISBN13', how='inner')

# COMMAND ----------

df_text_image_dataset['isbn'] = df_text_image_dataset['isbn'].apply(lambda x: x + ".jpg")
df_text_image_dataset = df_text_image_dataset[['isbn', 'description', 'is_high_sales_volume']]

# COMMAND ----------

df_text_image_dataset

# COMMAND ----------

# Filter the DataFrame to only include 3000 samples for each class
class_0_samples = df_text_image_dataset[df_text_image_dataset['is_high_sales_volume'] == 0].sample(n=7000, random_state=42)
class_1_samples = df_text_image_dataset[df_text_image_dataset['is_high_sales_volume'] == 1].sample(n=7000, random_state=42)
df = pd.concat([class_0_samples, class_1_samples])

df['isbn'] = df['isbn'].astype(str)
df['is_high_sales_volume'] = df['is_high_sales_volume'].astype(str)

# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# COMMAND ----------

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from transformers import BertTokenizer


# COMMAND ----------

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define function to load image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img



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

# Define image model with InceptionV3
# image_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(128, 128, 3))

# image_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
# image_model.trainable = False
# x = image_model.output
# x = layers.GlobalAveragePooling2D()(x)
# image_model = tf.keras.Model(inputs=image_model.input, outputs=x)

# # Define text model with LSTM
# text_input = layers.Input(shape=(512,), dtype='int32')
# text_embedding = layers.Embedding(len(tokenizer.get_vocab()), 128)(text_input)
# text_lstm = layers.LSTM(128)(text_embedding)


# Define image model with InceptionV3
image_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
image_model.trainable = False
x = image_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x) # Add an additional dense layer for image model enrichment
image_model = tf.keras.Model(inputs=image_model.input, outputs=x)

# Define text model with LSTM
text_input = layers.Input(shape=(512,), dtype='int32')
text_embedding = layers.Embedding(len(tokenizer.get_vocab()), 128)(text_input)
text_lstm = layers.LSTM(128)(text_embedding)
text_lstm = layers.Dense(256, activation='relu')(text_lstm) 

# Concatenate image and text models
concatenated = layers.concatenate([image_model.output, text_lstm])
dense1 = layers.Dense(512, activation='relu')(concatenated)
dense1 = layers.Dense(256, activation='relu')(dense1) 
output = layers.Dense(1, activation='sigmoid')(dense1)
model = tf.keras.models.Model(inputs=[image_model.input, text_input], outputs=output)




# COMMAND ----------

# Concatenate image and text models
concatenated = layers.concatenate([image_model.output, text_lstm])
dense1 = layers.Dense(512, activation='relu')(concatenated)
output = layers.Dense(1, activation='sigmoid')(dense1)
model = tf.keras.models.Model(inputs=[image_model.input, text_input], outputs=output)

# Compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# COMMAND ----------



# COMMAND ----------

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up checkpoint path
checkpoint_path = "/dbfs/team_j/text_image_combined_model_roger_20230427_InceptionV3"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# train
history = model.fit([train_images, train_texts], train_labels, validation_split=0.2, batch_size=32, epochs=10, callbacks=[cp_callback])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # model 2 - thicker text model layers (without lstm, use cnn instead , more dataset)

# COMMAND ----------

# Filter the DataFrame to only include 3000 samples for each class
class_0_samples = df_text_image_dataset[df_text_image_dataset['is_high_sales_volume'] == 0].sample(n=9000, random_state=42)
class_1_samples = df_text_image_dataset[df_text_image_dataset['is_high_sales_volume'] == 1].sample(n=9000, random_state=42)
df = pd.concat([class_0_samples, class_1_samples])

df['isbn'] = df['isbn'].astype(str)
df['is_high_sales_volume'] = df['is_high_sales_volume'].astype(str)

# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.001, random_state=42)

# COMMAND ----------



# COMMAND ----------

# train_ = pd.concat([train, test], axis=0)


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

from tensorflow.keras.applications import VGG16

# COMMAND ----------

# # Define image model with VGG16
# image_model = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
# image_model.trainable = False
# x = image_model.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(256, activation='relu')(x) # Add an additional dense layer for image model enrichment
# image_model = tf.keras.Model(inputs=image_model.input, outputs=x)

image_model = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
image_model.trainable = False
x = image_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x) 
x = layers.Dense(256, activation='relu')(x) # Add another dense layer for image model enrichment
x = layers.Dense(128, activation='relu')(x) # Add another dense layer for image model enrichment
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

# train_labels

# COMMAND ----------

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up checkpoint path
checkpoint_path = "/dbfs/team_j/text_image_combined_model_roger_20230427_vgg16_cnn_for_text_epoch5_trial_2"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# train
history = model.fit([train_images, train_texts], train_labels, validation_split=0.2, batch_size=32, epochs=5, callbacks=[cp_callback])

# COMMAND ----------

!ls /tmp/tmpqowhmbie/model/data/model/assets

# COMMAND ----------

! ls /dbfs/team_j/

# COMMAND ----------

! ls /dbfs/team_j/text_image_combined_model_roger_20230427_vgg16_cnn_for_text_epoch5_trial_2

# COMMAND ----------

!pip install --upgrade protobuf

# COMMAND ----------

!pip install --upgrade tensorflow

# COMMAND ----------

pip install --upgrade keras

# COMMAND ----------

!pip uninstall protobuf

# COMMAND ----------

!pip install protobuf

# COMMAND ----------

import mlflow.keras
import numpy as np

# Load the registered model
model_uri = 'runs:/5b5cda147afb42388d4aaeea222b46c1/model'
loaded_model = mlflow.keras.load_model(model_uri)

# Prepare some sample input data
image_data = np.random.rand(1, 128, 128, 3)
text_data = np.random.randint(0, 100, size=(1, 512))

# Make predictions using the loaded model
predictions = loaded_model.predict([image_data, text_data])

# Print the predictions
print(predictions)

# COMMAND ----------

loaded_model

# COMMAND ----------

from tensorflow.keras.models import load_model

# COMMAND ----------


# Define the path to the saved model
model_path = "/dbfs/team_j/text_image_combined_model_roger_20230427_vgg16_cnn_for_text_epoch5_trial_2"

# Load the model
model = load_model(model_path)

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct, col
logged_model = 'runs:/5b5cda147afb42388d4aaeea222b46c1/model'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

# Predict on a Spark DataFrame.
# df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))

# COMMAND ----------

loaded_model

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

model

# COMMAND ----------

model.save('/dbfs/team_j/text_image_combined_model_roger_20230427_vgg16_cnn_for_text_epoch5_trial_2.h5')

# COMMAND ----------

!cat /tmp/tmpqowhmbie/model/data/model/assets

# COMMAND ----------

train

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md # playground below

# COMMAND ----------

len(train),len(test)

# COMMAND ----------

# Sort values by 'isbn'
test = test.sort_values('isbn')
train = train.sort_values('isbn')

# COMMAND ----------

train = train.set_index('isbn')
test = test.set_index('isbn')

# COMMAND ----------



# COMMAND ----------

import numpy as np

def get_missing(file, df):
  parts = file.split(os.sep)
  idx = parts[-1]
  cls = parts[-2]
  indexes = df[:,0]
  classes = df[:,2]

  if idx in indexes:
    text = df[idx == indexes][0,1]
    return pd.NA, pd.NA, pd.NA
  else:
    text = df[cls == classes][0,1]
    
  return idx, text, cls   

vec_get_missing = np.vectorize(get_missing, signature='(),(m,n)->(),(),()')  

# COMMAND ----------

!ls '/dbfs/team_j/image_dataset/'

# COMMAND ----------

# train.index

# COMMAND ----------



# COMMAND ----------

parts

# COMMAND ----------


# Function for images loading

def add_not_found(path, df):
  files = glob.glob(path)
  df = df.reset_index()
  #idxs, texts, cls = vec_get_missing(files, df.values)
  #
  #found = pd.DataFrame({"description": texts,
  #                      "is_high_sales_volume": cls,
  #                     "isbn": idxs})
  #na = found.isna().sum().values[0]
  #if na<found.shape[0]:
  #  df = df.append(found)
  #df = df.drop_duplicates(subset='isbn', keep='first').dropna()
  df = df.set_index('isbn')
  #df = shuffle(df, random_state = 0)
  return df      

# Images folders 
train = add_not_found('/dbfs/team_j/image_dataset/*', train)
test = add_not_found('/dbfs/team_j/image_dataset/.*', test)

print("Number of training images:",train.shape[0])
print("Number of test images:",test.shape[0])

# COMMAND ----------

# df['text'][image_name]

# df = df.set_index('isbn')

df['description']

# COMMAND ----------

# Import the BERT BASE model from Tensorflow HUB (layer, vocab_file and tokenizer)
BertTokenizer = tokenization.tokenizer.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

# COMMAND ----------

# 

# COMMAND ----------

# Preprocessing of texts according to BERT +
# Cleaning of the texts

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence

def remove_tags(text):
    return TAG_RE.sub('', text)

TAG_RE = re.compile(r'<[^>]+>')
vec_preprocess_text = np.vectorize(preprocess_text)

def get_tokens(text, tokenizer):
  tokens = tokenizer.tokenize(text)
  tokens = ["[CLS]"] + tokens + ["[SEP]"]
  length = len(tokens)
  if length > max_length:
      tokens = tokens[:max_length]
  return tokens, length  

def get_masks(text, tokenizer, max_length):
    """Mask for padding"""
    tokens, length = get_tokens(text, tokenizer)
    return np.asarray([1]*len(tokens) + [0] * (max_length - len(tokens)))
vec_get_masks = np.vectorize(get_masks, signature = '(),(),()->(n)')

def get_segments(text, tokenizer, max_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    tokens, length = get_tokens(text, tokenizer)
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.asarray(segments + [0] * (max_length - len(tokens)))
vec_get_segments = np.vectorize(get_segments, signature = '(),(),()->(n)')

def get_ids(text, tokenizer, max_length):
    """Token ids from Tokenizer vocab"""
    tokens, length = get_tokens(text, tokenizer)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = np.asarray(token_ids + [0] * (max_length-length))
    return input_ids
vec_get_ids = np.vectorize(get_ids, signature = '(),(),()->(n)')

def get_texts(path):
    path = path.decode('utf-8')
    parts = path.split(os.sep)
    image_name = parts[-1]
    is_train = parts[-3] == 'train'
    if is_train:
      df = train
    else:
      df = test

    text = df['description'][image_name]
    return text
vec_get_text = np.vectorize(get_texts)
def prepare_text(paths):
    #Preparing texts
    
    texts = vec_get_text(paths)
    
    text_array = vec_preprocess_text(texts)
    ids = vec_get_ids(text_array, 
                      tokenizer, 
                      max_length).squeeze().astype(np.int32)
    masks = vec_get_masks(text_array,
                          tokenizer,
                          max_length).squeeze().astype(np.int32)
    segments = vec_get_segments(text_array,
                                tokenizer,
                                max_length).squeeze().astype(np.int32)
    
    return ids, segments, masks

def clean(i, tokens):
  try:
    this_token = tokens[i]
    next_token = tokens[i+1]
  except:
    return tokens
  if '##' in next_token:
      tokens.remove(next_token)
      tokens[i] = this_token + next_token[2:]
      tokens = clean(i, tokens)
      return tokens
  else:
    i = i+1
    tokens = clean(i, tokens)
    return tokens

def clean_text(array):
  array = array[(array!=0) & (array != 101) & (array != 102)]
  tokens = tokenizer.convert_ids_to_tokens(array)
  tokens = clean(0, tokens)
  text = ' '.join(tokens)
  return text

# COMMAND ----------

# Images preprocessing
def load_image(path):
    path = path.decode('utf-8')
    image = cv2.imread(path)
    image = cv2.resize(image, (img_width, img_height))
    image = image/255
    image = image.astype(np.float32)
    parts = path.split(os.sep)
    labels = parts[-2] == Classes 
    labels = labels.astype(np.int32)
    
    return image, labels
    
vec_load_image = np.vectorize(load_image, signature = '()->(r,c,d),(s)')

# COMMAND ----------

# Dataset creation

def prepare_data(paths):
    #Images and labels
    images, labels = tf.numpy_function(vec_load_image, 
                                      [paths], 
                                      [tf.float32, 
                                        tf.int32])
    
    
    [ids, segments, masks, ] = tf.numpy_function(prepare_text, 
                                              [paths], 
                                              [tf.int32, 
                                               tf.int32,
                                               tf.int32])
    images.set_shape([None, img_width, img_height, depth])
    labels.set_shape([None, nClasses])
    ids.set_shape([None, max_length])
    masks.set_shape([None, max_length])
    segments.set_shape([None, max_length])
    return ({"input_word_ids": ids, 
             "input_mask": masks,  
             "segment_ids": segments, 
             "image": images},
            {"class": labels})
    

    return dataset

# COMMAND ----------

# Parameters setting: images width and height, depth, number if classes, input shape
batch_size =  80
img_width = 128
img_height = 128
depth = 3
max_length = 20 #Setup according to the text

nClasses = train.is_high_sales_volume.nunique()
Classes = train.is_high_sales_volume.unique()
input_shape = (img_width, img_height, depth)

# COMMAND ----------

# Images loading using tf.data
def tf_data(path, batch_size):
    paths = tf.data.Dataset.list_files(path)
    paths = paths.batch(64)
    dataset = paths.map(prepare_data, tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset    
data_train = tf_data('/dbfs/team_j/image_dataset/*.jpg', batch_size)
data_test = tf_data('/dbfs/team_j/image_dataset/*.jpg', batch_size)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

def prepare_text(paths):
    #Preparing texts
    
    texts = vec_get_text(paths)
    
    text_array = vec_preprocess_text(texts)
    ids = vec_get_ids(text_array, 
                      tokenizer, 
                      max_length).squeeze().astype(np.int32)
    masks = vec_get_masks(text_array,
                          tokenizer,
                          max_length).squeeze().astype(np.int32)
    segments = vec_get_segments(text_array,
                                tokenizer,
                                max_length).squeeze().astype(np.int32)
    
    return ids, segments, masks

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# train

# COMMAND ----------

# !pip install bert-for-tf2

# COMMAND ----------

# !pip install bert-tokenizer

# COMMAND ----------

