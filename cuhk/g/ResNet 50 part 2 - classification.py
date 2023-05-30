# Databricks notebook source
pip install opencv-python

# COMMAND ----------

class_mapping = {
    0: 'graphic_print',
    1: 'multi_color',
    2: 'word_print',
    3: 'plain',
    4: 'checks',
    5: 'stripe',
    6: 'untagged',
    7: 'additional checks',
    8: 'additional diamonds'
}

# COMMAND ----------

class_mapping2 = {
    'graphic_print':0,
    'multi_color':1,
    'word_print':2,
    'plain':3,
    'checks':4,
    'stripe':5,
    'untagged':6
}

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import cv2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import tensorflow.keras as K 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale
from sklearn import preprocessing

def img_path(atg_code):
    return f"/dbfs/image/{atg_code}_in_xl.jpg"

def load_image(atg_code, resized_fac = 0.4):
    img     = cv2.imread(img_path(atg_code))
    h, _, _ = img.shape
    img   = img[0:int(h*2/3), :, :] # preserve upper half of the image only
    h, w, _ = img.shape
    resized = cv2.resize(img, (int(w*resized_fac), int(h*resized_fac)), interpolation = cv2.INTER_AREA)
    return resized

def load_image_simple(atg_code, resized_fac = 0.2):
    img     = cv2.imread(img_path(atg_code))
    return img

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

# DBTITLE 1,Image processing - import and cleansing embedding
# importing resampled image
df_train = pd.read_csv(os.path.join("file:/dbfs", "final_train_data3.csv"))
df_test = pd.read_csv(os.path.join("file:/dbfs", "final_test_data3.csv"))
df_val = pd.read_csv(os.path.join("file:/dbfs", "final_test_data3.csv"))

df_train.shape, df_test.shape, df_val.shape

# COMMAND ----------

# reading embedding
# df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new.csv"))
# df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new_50percent_upper.csv"))
# df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new_80percent.csv"))
# df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_restnet101.csv"))
df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_restnet152.csv"))

df_embs_original.groupby('Y').size()

# COMMAND ----------


X_train_amend = df_train[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1]
X_train = X_train_amend.to_numpy()
y_train = K.utils.to_categorical(df_train['class'])

X_test = df_test[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
y_test = K.utils.to_categorical(df_test['class'])

X_val = df_val[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
y_val = K.utils.to_categorical(df_val['class'])

X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape

# COMMAND ----------

#normalize data
from sklearn import preprocessing

normalizer = preprocessing.Normalizer()
normalized_train_X = normalizer.fit_transform(X_train)

normalized_test_X = normalizer.transform(X_test)
normalized_val_X = normalizer.transform(X_val)

normalized_train_X.shape, normalized_test_X.shape, normalized_val_X.shape

# COMMAND ----------

# DBTITLE 1,ResNet 152
from sklearn.metrics import accuracy_score
from sklearn import metrics

FILEPATH = "/dbfs/image_model_rest152"

EPOCH_SIZE = 50

# Early stopping  
check_point = [
    #K.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_SIZE/10),
    K.callbacks.ModelCheckpoint(filepath=FILEPATH, monitor="val_loss", mode="min", save_best_only=True,)
]

# Define neural network architecture
input_shape = (normalized_train_X.shape[1],)
num_classes = y_train.shape[1]

model = K.models.Sequential()
model.add(K.layers.Dense(256, input_shape = input_shape, activation="relu"))
#model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(normalized_train_X, y_train, batch_size=16, epochs=EPOCH_SIZE, verbose=1, validation_data=(normalized_val_X, y_val),callbacks=[check_point])

print(len(history.history['val_loss']))

model.summary()
model.save(FILEPATH)

# Evaluate the model on test data
y_pred = model.predict(normalized_test_X)
y_pred_class = np.argmax(y_pred, axis=1)

# consusion matrix
confusion_matrix = metrics.confusion_matrix(df_test['class'], y_pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# COMMAND ----------

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# COMMAND ----------

# DBTITLE 1,RenNet 101
# loading data 
df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_restnet101.csv"))

# loading training and testing data
X_train_amend = df_train[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1]
X_train = X_train_amend.to_numpy()
y_train = K.utils.to_categorical(df_train['class'])

X_test = df_test[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
y_test = K.utils.to_categorical(df_test['class'])

X_val = df_val[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
y_val = K.utils.to_categorical(df_val['class'])

X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape

#normalize data
normalizer = preprocessing.Normalizer()
normalized_train_X = normalizer.fit_transform(X_train)

normalized_test_X = normalizer.transform(X_test)
normalized_val_X = normalizer.transform(X_val)

normalized_train_X.shape, normalized_test_X.shape, normalized_val_X.shape

# COMMAND ----------

from sklearn.metrics import accuracy_score
from sklearn import metrics

FILEPATH = "/dbfs/image_model_rest101_new"

EPOCH_SIZE = 50

# Early stopping  
check_point = [
    K.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_SIZE/10),
    K.callbacks.ModelCheckpoint(filepath=FILEPATH, monitor="val_loss", mode="min", save_best_only=True,)
]

# Define neural network architecture
input_shape = (normalized_train_X.shape[1],)
num_classes = y_train.shape[1]

model = K.models.Sequential()
model.add(K.layers.Dense(32, input_shape = input_shape, activation="relu"))
#model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(normalized_train_X, y_train, batch_size=16, epochs=EPOCH_SIZE, verbose=1, validation_data=(normalized_val_X, y_val),callbacks=[check_point])

print(len(history.history['val_loss']))

model.summary()
model.save(FILEPATH)

# Evaluate the model on test data
y_pred = model.predict(normalized_test_X)
y_pred_class = np.argmax(y_pred, axis=1)

# consusion matrix
confusion_matrix = metrics.confusion_matrix(df_test['class'], y_pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# COMMAND ----------

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# COMMAND ----------

# DBTITLE 1,ResNet50 - no additional data
# loading data 
df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new_50percent_upper.csv"))

X_train_amend = df_train[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1]
X_train = X_train_amend.to_numpy()
y_train = K.utils.to_categorical(df_train['class'])

X_test = df_test[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
y_test = K.utils.to_categorical(df_test['class'])

X_val = df_val[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
y_val = K.utils.to_categorical(df_val['class'])

X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape

#normalize data
normalizer = preprocessing.Normalizer()
normalized_train_X = normalizer.fit_transform(X_train)

normalized_test_X = normalizer.transform(X_test)
normalized_val_X = normalizer.transform(X_val)

normalized_train_X.shape, normalized_test_X.shape, normalized_val_X.shape

# COMMAND ----------

from sklearn.metrics import accuracy_score
from sklearn import metrics

FILEPATH = "/dbfs/image_model_rest50_new_orginal_data"

EPOCH_SIZE = 50

# Early stopping  
check_point = [
    K.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_SIZE/10),
    K.callbacks.ModelCheckpoint(filepath=FILEPATH, monitor="val_loss", mode="min", save_best_only=True,)
]

# Define neural network architecture
input_shape = (normalized_train_X.shape[1],)
num_classes = y_train.shape[1]

model = K.models.Sequential()
model.add(K.layers.Dense(32, input_shape = input_shape, activation="relu"))
#model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(normalized_train_X, y_train, batch_size=16, epochs=EPOCH_SIZE, verbose=1, validation_data=(normalized_val_X, y_val),callbacks=[check_point])

print(len(history.history['val_loss']))

model.summary()
model.save(FILEPATH)

# Evaluate the model on test data
y_pred = model.predict(normalized_test_X)
y_pred_class = np.argmax(y_pred, axis=1)

# consusion matrix
confusion_matrix = metrics.confusion_matrix(df_test['class'], y_pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# COMMAND ----------

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,ResNet 50 with additional data
# loading data 
df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new_50percent_upper.csv"))

# adding additional check image
df_image_file = pd.read_csv(os.path.join("file:/dbfs", "image_file_name.csv"))
df_image_file['Y'] = df_image_file['Y']-1
df_image_file = df_image_file.loc[df_image_file['Y']!=4]
df_image_file.loc[df_image_file["Y"] == 7, "Y"] = 4 # 8 is check from online
df_check = df_image_file.loc[(df_image_file['Y']==4)][['atg_code']]

df_train_new = df_train.loc[(df_train['class']!=4)][['atg_code']].append(df_check.sample(n=300))
X_train_amend = df_train_new.merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1]
y_train = K.utils.to_categorical(df_train_new.merge(df_image_file, on=['atg_code'], how='inner').iloc[:,-1:])

#X_train_amend = df_train[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1]
X_train = X_train_amend.to_numpy()
#y_train = K.utils.to_categorical(df_train['class'])

X_test = df_test[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
y_test = K.utils.to_categorical(df_test['class'])

X_val = df_val[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
y_val = K.utils.to_categorical(df_val['class'])

X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape

#normalize data
normalizer = preprocessing.Normalizer()
normalized_train_X = normalizer.fit_transform(X_train)

normalized_test_X = normalizer.transform(X_test)
normalized_val_X = normalizer.transform(X_val)

normalized_train_X.shape, normalized_test_X.shape, normalized_val_X.shape

# COMMAND ----------

from sklearn.metrics import accuracy_score
from sklearn import metrics

FILEPATH = "/dbfs/image_model_rest50_new"

EPOCH_SIZE = 50

# Early stopping  
check_point = [
    K.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_SIZE/10),
    K.callbacks.ModelCheckpoint(filepath=FILEPATH, monitor="val_loss", mode="min", save_best_only=True,)
]

# Define neural network architecture
input_shape = (normalized_train_X.shape[1],)
num_classes = y_train.shape[1]

model = K.models.Sequential()
model.add(K.layers.Dense(32, input_shape = input_shape, activation="relu"))
#model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(normalized_train_X, y_train, batch_size=16, epochs=EPOCH_SIZE, verbose=1, validation_data=(normalized_val_X, y_val),callbacks=[check_point])

print(len(history.history['val_loss']))

model.summary()
model.save(FILEPATH)

# Evaluate the model on test data
y_pred = model.predict(normalized_test_X)
y_pred_class = np.argmax(y_pred, axis=1)

# consusion matrix
confusion_matrix = metrics.confusion_matrix(df_test['class'], y_pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# COMMAND ----------

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# COMMAND ----------



# COMMAND ----------

