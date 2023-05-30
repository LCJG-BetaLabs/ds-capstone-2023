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

# show image path
import pyspark.dbutils
image_path_list = dbutils.fs.ls("file:/dbfs/FileStore/tables/image/check/")
image_path_list

# COMMAND ----------

for i in range(30, 301):
    dbutils.fs.cp(f"file:/dbfs/FileStore/tables/image/check/check_{i}.jpg", f"file:/dbfs/image/check_{i}.jpg")

# COMMAND ----------

# show image path
import pyspark.dbutils
image_path_list = dbutils.fs.ls("file:/dbfs/image/")
image_path_list[:5]

# COMMAND ----------

def img_path(atg_code):
    return f"/dbfs/image/{atg_code}_in_xl.jpg"

def load_image(atg_code, resized_fac = 0.2):
    img     = cv2.imread(img_path(atg_code))
    h, _, _ = img.shape
    img   = img[0:int(h*2/3), :, :] # preserve upper half of the image only
    h, w, _ = img.shape
    resized = cv2.resize(img, (int(w*resized_fac), int(h*resized_fac)), interpolation = cv2.INTER_AREA)
    return resized

def load_image_original(atg_code, resized_fac = 0.2):
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

# DBTITLE 1,Use Pre-Trained Model to Recommendation
# show image and how we cut the image in half
img_array = load_image("BWJ579")
img_array2 = load_image_original("BWJ579")

plt.figure()
f, axarr = plt.subplots(1,2) 
axarr[0].imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
axarr[1].imshow(cv2.cvtColor(img_array2, cv2.COLOR_BGR2RGB))

# COMMAND ----------

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

# length of embedding must be 2048
emb = get_embedding(model, "BUE682")
emb.shape[0]

# COMMAND ----------

emb.shape

# COMMAND ----------

img_array = load_image("BUE682")
plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
print(img_array.shape)
print(emb)

# COMMAND ----------

import pandas as pd
import os
df = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df.head()

# COMMAND ----------

df_sample      = df
map_embeddings = df_sample['atg_code'].apply(lambda img: get_embedding(model, img))
df_embs        = map_embeddings.apply(pd.Series)

print(df_embs.shape)
df_embs.to_csv('/dbfs/image_embedding_data.csv')
df_embs.head()

# COMMAND ----------

df_embs['Y'] = df['class']

# COMMAND ----------

df_embs.to_csv('/dbfs/image_embedding_data.csv')

# COMMAND ----------

from sklearn.metrics.pairwise import pairwise_distances
cosine_sim = 1-pairwise_distances(df_embs, metric='cosine')
cosine_sim[:4, :4]

# COMMAND ----------

indices = pd.Series(range(len(df)), index=df.index)
indices

# Function that get movie recommendations based on the cosine similarity score of movie genres
def get_recommender(idx, df, top_n = 5):
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_sim[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
    return indices.iloc[idx_rec].index, idx_sim

get_recommender(333, df, top_n = 5)

# COMMAND ----------

# Idx Item to Recommender
idx_ref = 856

# Recommendations
idx_rec, idx_sim = get_recommender(idx_ref, df, top_n = 6)

# Plot
#===================
print('im'+str(idx_ref)+f' class: {df.iloc[idx_ref]["class"]}')
plt.imshow(cv2.cvtColor(load_image(df.iloc[idx_ref].atg_code), cv2.COLOR_BGR2RGB))

# generation of a dictionary of (title, images)
figures = {f'im:{str(i)} class:{df.iloc[i]["class"]} \nsim:{sim}': load_image(row.atg_code) for (i, row), sim in zip(df.loc[idx_rec].iterrows(),idx_sim)}
# plot of the images in a figure, with 2 rows and 3 columns
plot_figures(figures, 2, 3)

# COMMAND ----------

df_random = df.loc[df['class']==7].sample(n=6)
# generate image of same class
figures = {f'im:{str(i)} class:{df.iloc[i]["class"]}': load_image(row.atg_code) for i, row in df_random.iterrows()}
# plot of the images in a figure, with 2 rows and 3 columns
plot_figures(figures, 2, 3)
# text info
df_random
df_random['long_desc']

# COMMAND ----------

#it appears that the result is not related to pattern similarity

# COMMAND ----------

import tensorflow.keras as K 
def preprocess_data(X, Y, NUM_CLASSES):
    """
    a function that trains a convolutional neural network to classify the
    CIFAR 10 dataset
    :param X: X is a numpy.ndarray of shape (m, 32, 32, 3) containing the
    CIFAR 10 data, where m is the number of data points
    :param Y: Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
    labels for X
    :cat_num: number of categories 
    :return: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=NUM_CLASSES)
    return X_p, Y_p

# COMMAND ----------

import numpy as np
X = np.array([load_image(i) for i in df.loc[df['class']!=7].atg_code]) # remove untagged images
y = np.array(df.loc[df['class']!=7]['class'].tolist()) # remove untagged images
print(X.shape, y.shape)

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# COMMAND ----------

X_train, y_train = preprocess_data(X_train, y_train, 6+1)
X_test, y_test = preprocess_data(X_test, y_test, 6+1)
print((X_train.shape, y_train.shape))

# COMMAND ----------

FILEPATH = "/dbfs/model.v1"

img_width, img_height, color = load_image("BUE682").shape # assume all images have the same shape

# Trained Model
new_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape = (img_width, img_height, color))

print(img_width, img_height)

for layer in new_model.layers:
    layer.trainable = False
# Check the freezed was done ok
for i, layer in enumerate(new_model.layers):
    print(i, layer.name, "-", layer.trainable)

to_res = (330, 360)

model = K.models.Sequential()
model.add(K.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
model.add(new_model)
model.add(K.layers.Flatten())
model.add(K.layers.Dense(256, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(7, activation='softmax'))

check_point = K.callbacks.ModelCheckpoint(filepath=FILEPATH,
                                              monitor="val_acc",
                                              mode="max",
                                              save_best_only=True,
                                              )

model.compile(loss='categorical_crossentropy',
              optimizer=K.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=8, epochs=10, verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=[check_point])

model.summary()
model.save(FILEPATH)
# best model batch size 32
# flat -> normalize 

# 256 - highest val_accuracy 0.71. it overfit after 3 epoch

# COMMAND ----------

# DBTITLE 1,version 2
FILEPATH = "/dbfs/model.v2"

img_width, img_height, color = load_image("BUE682").shape # assume all images have the same shape

# Trained Model
new_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape = (img_width, img_height, color))

print(img_width, img_height)

for layer in new_model.layers:
    layer.trainable = False
# Check the freezed was done ok
for i, layer in enumerate(new_model.layers):
    print(i, layer.name, "-", layer.trainable)

to_res = (330, 360)

model = K.models.Sequential()
model.add(K.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
model.add(new_model)
model.add(K.layers.Flatten())
model.add(K.layers.Dense(128, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(7, activation='softmax'))

check_point = K.callbacks.ModelCheckpoint(filepath=FILEPATH,
                                              monitor="val_acc",
                                              mode="max",
                                              save_best_only=True,
                                              )

model.compile(loss='categorical_crossentropy',
              optimizer=K.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=8, epochs=10, verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=[check_point])

model.summary()
model.save(FILEPATH)

# COMMAND ----------

FILEPATH = "/dbfs/model.v3"

img_width, img_height, color = load_image("BUE682").shape # assume all images have the same shape

# Trained Model
new_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape = (img_width, img_height, color))

print(img_width, img_height)

for layer in new_model.layers:
    layer.trainable = False
# Check the freezed was done ok
for i, layer in enumerate(new_model.layers):
    print(i, layer.name, "-", layer.trainable)

to_res = (330, 360)

model = K.models.Sequential()
model.add(K.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
model.add(new_model)
model.add(K.layers.Flatten())
model.add(K.layers.Dense(512, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(7, activation='softmax'))

check_point = K.callbacks.ModelCheckpoint(filepath=FILEPATH,
                                              monitor="val_acc",
                                              mode="max",
                                              save_best_only=True,
                                              )

model.compile(loss='categorical_crossentropy',
              optimizer=K.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=8, epochs=10, verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=[check_point])

model.summary()
model.save(FILEPATH)

# COMMAND ----------

FILEPATH = "/dbfs/model.v3"

img_width, img_height, color = load_image("BUE682").shape # assume all images have the same shape

# Trained Model
new_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape = (img_width, img_height, color))

print(img_width, img_height)

for layer in new_model.layers:
    layer.trainable = False
# Check the freezed was done ok
for i, layer in enumerate(new_model.layers):
    print(i, layer.name, "-", layer.trainable)

to_res = (330, 360)

model = K.models.Sequential()
model.add(K.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
model.add(new_model)
model.add(K.layers.Flatten())
model.add(K.layers.Dense(256, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(128, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(7, activation='softmax'))

check_point = K.callbacks.ModelCheckpoint(filepath=FILEPATH,
                                              monitor="val_acc",
                                              mode="max",
                                              save_best_only=True,
                                              )

model.compile(loss='categorical_crossentropy',
              optimizer=K.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=8, epochs=10, verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=[check_point])

model.summary()
model.save(FILEPATH)

# COMMAND ----------

