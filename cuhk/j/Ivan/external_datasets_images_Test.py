# Databricks notebook source
# MAGIC %md # Two seperate runs that fetches books details from google api

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm_20230421_2")
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_4.json").createOrReplaceTempView("isbn_google_reomm_20230421_4")

# COMMAND ----------

# MAGIC %md # get isbn and image link pair (excluding rows with null image link )

# COMMAND ----------

# MAGIC %md #### 32071 image links found from google api

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded_2 as (
# MAGIC   select 
# MAGIC       explode(items)
# MAGIC   from isbn_google_reomm_20230421_2
# MAGIC ),
# MAGIC volumeinfo_2 AS (
# MAGIC   select 
# MAGIC   col.volumeInfo.*
# MAGIC   FROM
# MAGIC       exploded_2
# MAGIC ),
# MAGIC exploded_4 as (
# MAGIC   select 
# MAGIC       explode(items)
# MAGIC   from isbn_google_reomm_20230421_4
# MAGIC ),
# MAGIC volumeinfo_4 AS (
# MAGIC   select 
# MAGIC   col.volumeInfo.*
# MAGIC   FROM
# MAGIC       exploded_2
# MAGIC ),
# MAGIC unioned AS (
# MAGIC   select * from volumeinfo_2
# MAGIC   union all 
# MAGIC   select * from volumeinfo_4
# MAGIC ),
# MAGIC isbn_image_link_pair AS (
# MAGIC   SELECT
# MAGIC     distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
# MAGIC     imageLinks.thumbnail as thumbnail
# MAGIC   FROM
# MAGIC     unioned
# MAGIC )
# MAGIC select 
# MAGIC   count(1)
# MAGIC from 
# MAGIC   isbn_image_link_pair
# MAGIC where 
# MAGIC   thumbnail is not null 

# COMMAND ----------

isbn_thumbnail_pairs = spark.sql("""
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
        imageLinks.thumbnail as thumbnail
    FROM
        unioned
    )
    select 
        *
    from 
    isbn_image_link_pair
    where 
    thumbnail is not null 
""").rdd.map(lambda x:{x['isbn']:x['thumbnail']}).collect()


# COMMAND ----------

for item in isbn_thumbnail_pairs[:5]:
    print(list(item.keys())[0])

# COMMAND ----------

!/databricks/python3/bin/python -m pip install --upgrade pip

# COMMAND ----------

!pip install asyncio aiohttp

# COMMAND ----------

from shutil import copyfile

# COMMAND ----------

import asyncio
import aiohttp
import os


# COMMAND ----------

target_dir = "./team_j_downloaded_images/raw_20230423/"

# Check if the directory exists
if os.path.exists(target_dir):
    print('Directory exists')

# COMMAND ----------

target_dir

# COMMAND ----------

import asyncio
import aiohttp
import os


target_dir = "./team_j_downloaded_images/raw_20230423/"

# Check if the directory exists
if os.path.exists(target_dir):
    print('Directory exists')
else:
    # Create the directory if it doesn't exist
    os.makedirs(target_dir)
    print(f'Directory {target_dir} created')


async def download_image(url, filename, retries=3):
    async with aiohttp.ClientSession() as session:
        while retries > 0:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(filename, 'wb') as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    return
                else:
                    print(f'Response status was {response.status}, {retries} retries left, retrying...')
                    retries -= 1
                    await asyncio.sleep(1)
        print(f'Failed to download {url} after {retries} retries.')

async def run():

    #url = 'http://books.google.com/books/content?id=vQKPswEACAAJ&printsec=frontcover&img=1&zoom=1&source=gbs_api'
    #filename = f'{target_dir}9781680522075.jpg'
    #await download_image(url, filename)
    #print(f'Saved {filename} successfully!')

    for item in isbn_thumbnail_pairs[:]:
        for isbn, url in item.items():
            #print(isbn, url)
            filename = f'{target_dir}{isbn}.jpg'
            await download_image(url, filename)
            print(f'Saved {filename} successfully!')

asyncio.run(run())



# COMMAND ----------


src_dir = target_dir
dest_dir = f'/dbfs/{src_dir.split("./")[1]}'

# Check if the directory exists
if os.path.exists(dest_dir):
    print(f'Directory {dest_dir} exists')
else:
    # Create the directory if it doesn't exist
    os.makedirs(dest_dir)
    print(f'Directory {dest_dir} created')


file_list = os.listdir(src_dir)

for file_name in file_list:
    # Construct the full file paths
    src_file = os.path.join(src_dir, file_name)
    dest_file = os.path.join(dest_dir, file_name)

    # Copy the file using shutil.copyfile()
    copyfile(src_file, dest_file)
    print(f"copied file {file_name} to file path : {dest_file}")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # inspect fetched images

# COMMAND ----------

!ls team_j_downloaded_images/raw_20230423 | wc -l

# COMMAND ----------

!ls /dbfs/team_j_downloaded_images/raw_20230423 | wc -l

# COMMAND ----------

# MAGIC %md #### only successfully 31747 images from the thumbnail links

# COMMAND ----------

# MAGIC %md ##### show a few images with isbn

# COMMAND ----------

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from IPython.core.display import HTML

from IPython.display import display

image_dir = "/dbfs/team_j_downloaded_images/raw_20230423/"
raw_image_list = os.listdir(f'{image_dir}')


file_paths = raw_image_list[:30]

# Calculate the number of rows and columns needed to display the images in a grid
num_images = len(file_paths)
num_cols = 5
num_rows = math.ceil(num_images / num_cols)

# Create a new figure with the appropriate size
fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, 20), gridspec_kw={'hspace': 0.4, 'wspace': 0})
# ax = ax.ravel()

# Loop through the file paths and display each image in a subplot
for i, file_path in enumerate(file_paths):
    row = i // num_cols
    col = i % num_cols
    img = mpimg.imread(f"{image_dir}{file_path}")
    title = file_path
    ax[row][col].imshow(img)
    ax[row][col].set_title(title)
    ax[row][col].axis('off')

# Hide any unused subplots
for i in range(num_images, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    ax[row][col].axis('off')

# Display the plot
html = HTML(f'<div style="overflow-x: scroll; width: 100%;">{fig}</div>')
plt.show()

# Display the plot
# fig.tight_layout()
# display(fig)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# COMMAND ----------

#!pip install opencv-python

# COMMAND ----------

import cv2

im = cv2.imread("/dbfs/team_j_downloaded_images/image_Class/LowSales/9780001374386.jpg.jpg")
print(type(im))
print(im.shape)""
print(type(im.shape))

# COMMAND ----------

raw_image_list[0]

# COMMAND ----------

for i in range(0, 100):
    try: 
        x = f"/dbfs/team_j_downloaded_images/image_Class/LowSales/{raw_image_list[i]}"
        im = cv2.imread(x)
        print(im.shape)
    except:
        None

# COMMAND ----------

def scale_resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # equivalent to dividing image pixels by 255
    image = tf.image.resize(image, (200, 128)) # Resizing the image to 224x224 dimention
    return (image, label)


# COMMAND ----------

img_height,img_width=200,128
batch_size=64
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

class_names = train_ds.class_names
print(class_names)

# COMMAND ----------

resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(200,128,3),
                   pooling='avg',classes=3,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(3, activation='softmax'))

# COMMAND ----------

resnet_model.summary()


# COMMAND ----------

# Set Optimizer
opt = Adam(lr=0.001)
resnet_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


# COMMAND ----------

epochs=10
history = resnet_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
  

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

PIL.Image.open(raw_image_list)

# COMMAND ----------

raw_image_list[0]

# COMMAND ----------



# COMMAND ----------

# !pip install opencv-python

# COMMAND ----------

!ls /dbfs/

# COMMAND ----------

# import cv2
# # image_path = "9781680522075.jpg"
# image_path = "/dbfs/team_j_downloaded_images/raw_20230423/9781680522075.jpg"

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Load an image from a file path
# img = mpimg.imread(image_path)

# # Display the image using pyplot
# plt.imshow(img)

# # Show the plot
# plt.show()

# COMMAND ----------

# !ls

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# dest_dir

# COMMAND ----------

# dbutils.fs.ls("/dbfs/")

# COMMAND ----------

# dest_dir

# COMMAND ----------

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
batch_size = 32
img_height = 180
img_width = 180

# COMMAND ----------

train_ds = tf.keras.utils.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

val_ds = tf.keras.utils.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

class_names = train_ds.class_names
print(class_names)

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# COMMAND ----------

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# COMMAND ----------

normalization_layer = tf.keras.layers.Rescaling(1./255)


# COMMAND ----------

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# COMMAND ----------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# COMMAND ----------

num_classes = 3

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# COMMAND ----------

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# COMMAND ----------

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

# COMMAND ----------



# COMMAND ----------

import matplotlib.pyplot as plotter_lib

import numpy as np

import PIL as image_lib

import tensorflow as tflow

from tensorflow.keras.layers import Flatten

from keras.layers.core import Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

# COMMAND ----------

img_height,img_width=180,180

batch_size=32

train_ds = tflow.keras.preprocessing.image_dataset_from_directory(

  f"/dbfs/team_j_downloaded_images/image_Class/",

  validation_split=0.2,

  subset="training",

  seed=123,

label_mode='categorical',

  image_size=(img_height, img_width),

  batch_size=batch_size)

# COMMAND ----------

validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(

  f"/dbfs/team_j_downloaded_images/image_Class/",

  validation_split=0.2,

  subset="validation",

  seed=123,

label_mode='categorical',

  image_size=(img_height, img_width),

  batch_size=batch_size)

# COMMAND ----------

import matplotlib.pyplot as plotter_lib

plotter_lib.figure(figsize=(10, 10))

epochs=10

for images, labels in train_ds.take(1):

  for var in range(6):

    ax = plt.subplot(3, 3, var + 1)

    plotter_lib.imshow(images[var].numpy().astype("uint8"))

    plotter_lib.axis("off")*



# COMMAND ----------

demo_resnet_model = Sequential()

pretrained_model_for_demo= tflow.keras.applications.ResNet50(include_top=False,

                   input_shape=(180,180,3),

                   pooling='avg',classes=3,

                   weights='imagenet')

for each_layer in pretrained_model_for_demo.layers:

        each_layer.trainable=False

demo_resnet_model.add(pretrained_model_for_demo)

demo_resnet_model.add(Flatten())

demo_resnet_model.add(Dense(512, activation='relu'))

demo_resnet_model.add(Dense(3, activation='softmax'))



# COMMAND ----------

demo_resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

history = demo_resnet_model.fit(train_ds, validation_data=validation_ds, epochs=epochs)



# COMMAND ----------

plotter_lib.figure(figsize=(8, 8))

epochs_range= range(epochs)

plotter_lib.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")

plotter_lib.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")

plotter_lib.axis(ymin=0.4,ymax=1)

plotter_lib.grid()

plotter_lib.title('Model Accuracy')

plotter_lib.ylabel('Accuracy')

plotter_lib.xlabel('Epochs')

plotter_lib.legend(['train', 'validation'])

# COMMAND ----------

#plotter_lib.show()

plotter_lib.savefig('output-plot.png') 

# COMMAND ----------

# Save model as .h5 file
demo_resnet_model.save('/dbfs/team_j/model_checkpoint_ivan_three_class_20230426_trial')

# COMMAND ----------

# Save model as .h5 file
demo_resnet_model.save('.model_checkpoint_ivan_three_class_20230426_trial')

# COMMAND ----------

import glob

glob.glob('/dbfs/team_j/*')

# COMMAND ----------

# Save model as .h5 file
demo_resnet_model.save('/dbfs/team_j/model_checkpoint_ivan_three_class_20230426_trial.h5')

# COMMAND ----------

dir(demo_resnet_model)

# COMMAND ----------


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

