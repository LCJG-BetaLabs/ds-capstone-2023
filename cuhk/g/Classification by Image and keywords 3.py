# Databricks notebook source
pip install opencv-python

# COMMAND ----------

#%sh apt-get -f -y install tesseract-ocr 

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
# df_train = pd.read_csv(os.path.join("file:/dbfs", "final_train_data.csv"))
# df_test = pd.read_csv(os.path.join("file:/dbfs", "final_test_data.csv"))
# df_val = pd.read_csv(os.path.join("file:/dbfs", "final_test_data.csv"))

# df_train = pd.read_csv(os.path.join("file:/dbfs", "final_train_data2.csv"))
# df_test = pd.read_csv(os.path.join("file:/dbfs", "final_test_data2.csv"))
# df_val = pd.read_csv(os.path.join("file:/dbfs", "final_test_data2.csv"))

df_train = pd.read_csv(os.path.join("file:/dbfs", "final_train_data3.csv"))
df_test = pd.read_csv(os.path.join("file:/dbfs", "final_test_data3.csv"))
df_val = pd.read_csv(os.path.join("file:/dbfs", "final_test_data3.csv"))

# df_train = pd.read_csv(os.path.join("file:/dbfs", "final_train_data4.csv"))
# df_test = pd.read_csv(os.path.join("file:/dbfs", "final_test_data4.csv"))
# df_val = pd.read_csv(os.path.join("file:/dbfs", "final_test_data4.csv"))

df_train.shape, df_test.shape, df_val.shape

# COMMAND ----------

# reading embedding
# df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new.csv"))

df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new_50percent_upper.csv"))
# df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new_80percent.csv"))
# df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_restnet101.csv"))
# df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_restnet152.csv"))

df_embs_original.groupby('Y').size()

# COMMAND ----------

# DBTITLE 1,increase sample size of check pattern
df_image_file = pd.read_csv(os.path.join("file:/dbfs", "image_file_name.csv"))
df_image_file['Y'] = df_image_file['Y']-1
df_image_file = df_image_file.loc[df_image_file['Y']!=4]
# df_image_file = df_image_file.loc[df_image_file['Y']!=5]
df_image_file.loc[df_image_file["Y"] == 7, "Y"] = 4 # 8 is check from online
# df_image_file.loc[df_image_file["Y"] == 8, "Y"] = 4 # 8 is diamond from online
# df_image_file.loc[df_image_file["Y"] == 9, "Y"] = 5 # 8 is check from online
df_check = df_image_file.loc[(df_image_file['Y']==4)][['atg_code']]

df_train_new = df_train.loc[(df_train['class']!=4)][['atg_code']].append(df_check.sample(n=300))
#df_train_new = df_train[['atg_code']].append(df_check_strips)
X_train_amend = df_train_new.merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1]
y_train = K.utils.to_categorical(df_train_new.merge(df_image_file, on=['atg_code'], how='inner').iloc[:,-1:])

# df_train_new.merge(df_image_file, on=['atg_code'], how='inner').iloc[:,-1:].groupby('Y').size()

# COMMAND ----------

# df_new_tag = pd.read_csv("/dbfs/FileStore/shared_uploads/capstone2023_cuhk_team_g@ijam.onmicrosoft.com/new_tag2a.csv")
# df_new_tag['tag_image'] = df_new_tag['tag'].map(class_mapping2)

# df_train_newtag = df_train[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').merge(df_new_tag, on=['atg_code'], how='inner')
# X_train = df_train_newtag.iloc[:, 1:-6].to_numpy()
# y_train = K.utils.to_categorical(df_train_newtag.iloc[:, -1:])

# COMMAND ----------


#X_train_amend = df_train[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1]
X_train = X_train_amend.to_numpy()
#y_train = K.utils.to_categorical(df_train['class'])

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

# DBTITLE 1,text embedding 
# # Text embedding
# df_embs_gpt = pd.read_csv(os.path.join("file:/dbfs", "final_gpt_ebd_df_without7.csv")).iloc[:, 1:]

# X_train_gpt = df_train[['atg_code']].merge(df_embs_gpt, on=['atg_code'], how='inner').iloc[:,2:].to_numpy()
# y_train = K.utils.to_categorical(df_train['class'])

# X_test_gpt = df_test[['atg_code']].merge(df_embs_gpt, on=['atg_code'], how='inner').iloc[:,2:].to_numpy()
# y_test = K.utils.to_categorical(df_test['class'])

# X_val_gpt = df_val[['atg_code']].merge(df_embs_gpt, on=['atg_code'], how='inner').iloc[:,2:].to_numpy()
# y_val = K.utils.to_categorical(df_val['class'])

# COMMAND ----------

# # text only
# normalized_train_X = X_train_gpt
# normalized_test_X = X_test_gpt
# normalized_val_X = X_val_gpt

# COMMAND ----------

# DBTITLE 1,Combined X
# normalized_train_X = np.concatenate((normalized_train_X, X_train_gpt), axis=1)
# normalized_test_X = np.concatenate((normalized_test_X, X_test_gpt), axis=1)
# normalized_val_X = np.concatenate((normalized_val_X, X_val_gpt), axis=1)

# COMMAND ----------

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

import matplotlib.pyplot as plt

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

from tensorflow import keras
model= keras.models.load_model('/dbfs/image_model_rest50')

# Evaluate the model on test data
y_pred = model.predict(normalized_test_X)
y_pred_class = np.argmax(y_pred, axis=1)

# consusion matrix
confusion_matrix = metrics.confusion_matrix(df_test['class'], y_pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# COMMAND ----------

df_prob_test = pd.concat([df_test[['atg_code']], pd.DataFrame(y_pred_class), pd.DataFrame(y_pred).reset_index(drop=True)], axis=1)
df_prob_test.columns = [['atg_code','pred',"Class1","Class2","Class3","Class4","Class5","Class6"]]
df_prob_test.to_csv('/dbfs/image_prob_test_rest50.csv')
df_prob_test.head(10)

# COMMAND ----------

df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df2 = df_cleansed.loc[df_cleansed['class']==7][['atg_code']]
df2

X_class7 = df2[['atg_code']].merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
normalized_test_X_class7 = normalizer.transform(X_class7)
y_pred_class7 = model.predict(normalized_test_X_class7)
y_pred_class7_prob = np.argmax(y_pred_class7, axis=1)

df2['pred'] = y_pred_class7_prob
df2.groupby('pred').size()

# COMMAND ----------

df_prob_test

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def display_image(path, dpi=200):
    """
    Description:
        Displayes an image
    Inputs:
        path (str): File path
        dpi (int): Your monitor's pixel density
    """
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize = (width/dpi,height/dpi))
    plt.imshow(img, interpolation='nearest', aspect='auto')

display_image(img_path("BWA955"))

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def display_image(path, dpi=200):
    """
    Description:
        Displayes an image
    Inputs:
        path (str): File path
        dpi (int): Your monitor's pixel density
    """
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize = (width/dpi,height/dpi))
    plt.imshow(img, interpolation='nearest', aspect='auto')

#df_view = df_word_count.loc[(df_word_count['word_length1']<10) & (df_word_count['word_length2']<50)] 
df_view = df_prob_test.loc[df_prob_test['pred']==4].reset_index()
print(len(df_view))
sample_n = 2
class_list = []

for idx, items in df_view.sample(n=sample_n).iterrows():
    #class_list.append(items['atg_code'])
    #print(df.iloc[idx,:])
    #print(items['atg_code'], items['word_length1'], items['word_length2'])
    display_image(img_path(items.atg_code))

# COMMAND ----------

df_prob_untagged = pd.concat([df2.reset_index(drop=True), pd.DataFrame(y_pred_class7).reset_index(drop=True)], axis=1)
df_prob_untagged.columns = [['atg_code','pred',"Class1","Class2","Class3","Class4","Class5","Class6"]]
df_prob_untagged.to_csv('/dbfs/image_prob_untagged_rest50.csv')
df_prob_untagged.head(10)

# COMMAND ----------

import cv2
import matplotlib.pyplot as plt

def img_path(atg_code):
    if atg_code[0:5] == 'check':
        return f"/dbfs/FileStore/tables/image/checks/{atg_code}.jpg"
    if atg_code[0:8] == 'diamonds':
        return f"/dbfs/FileStore/tables/image/diamonds/{atg_code}.jpg"
    else :
        return f"/dbfs/image/{atg_code}_in_xl.jpg"

def load_image_new(atg_code, resized_fac = 0.2):
    img     = cv2.imread(img_path(atg_code))
    h, _, _ = img.shape
    img   = img[0:int(h*2/3), :, :] # preserve upper half of the image only
    h, w, _ = img.shape
    resized = cv2.resize(img, (int(w*resized_fac), int(h*resized_fac)), interpolation = cv2.INTER_AREA)
    return resized

def load_image(atg_code, resized_fac = 0.2):
    if atg_code[0:5] != 'check':
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

img_array = load_image(img_path("BWL954"))
plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
print(img_array.shape)

# COMMAND ----------

display_image(img_path("BWQ714"))

# COMMAND ----------

# find text
import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt

df_word_length = pd.DataFrame(columns=['word_length1', 'word_length2'])
config1 = r'--psm 1 --oem 3 --dpi 300 -l eng'
config2 = r'--psm 6 --oem 1 -l eng'
for idx, items in df_cleansed.iterrows():
    img = load_image(items.atg_code)
    text = pytesseract.image_to_string(img, config=config1)
    text2 = pytesseract.image_to_string(img, config=config2)
    word_length = [[len(text.replace('\x0c', '')), len(text2.replace('\x0c', ''))]]
    df = pd.DataFrame(word_length, columns=['word_length1', 'word_length2'])
    df_word_length = df_word_length.append(df)
    print(idx)

df_word_length

# COMMAND ----------

df_word_length.to_csv('/dbfs/word_length_pred.csv')

# COMMAND ----------

df_word_length = df_word_length.reset_index(drop=True)

# COMMAND ----------

df_cleansed = df_cleansed.reset_index(drop=True)

# COMMAND ----------

df_word_count = pd.concat([df_cleansed, df_word_length], axis=1)

# COMMAND ----------

df_word_count.to_csv('/dbfs/word_length_pred.csv')

# COMMAND ----------

df_word_count = pd.read_csv(os.path.join("file:/dbfs", "word_length_pred.csv"))
df_word_count

# COMMAND ----------

df_word_count

# COMMAND ----------

atg_code = 'BVV864'
img = load_image(atg_code)

config1 = r'--psm 1 --oem 3 --dpi 300 -l eng'
config2 = r'--psm 6 --oem 1 -l eng'
text = pytesseract.image_to_string(img, config=config1)
print(len(text.replace('\x0c', '')))
text2 = pytesseract.image_to_string(img, config=config2)
print(len(text2.replace('\x0c', '')))
display_image(img_path(atg_code))

# a > 0
# b > 10
# b < 60

# COMMAND ----------

display_image(img_path('BXB418'))

# COMMAND ----------

print(df_cleansed.loc[df_cleansed['class']==5].shape[0] )
print(df_cleansed.loc[df_cleansed['class']==5]['atg_code'].iloc[14] )
df_cleansed.loc[df_cleansed['class']==5]['long_desc'].iloc[14] # 2 plaid diamanté boxy squared gingham checked argyle
# 5 6 7 12 fail
# 8 11 BVV864 box?

# COMMAND ----------

display_image(img_path_new('check_100'))

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def display_image(path, dpi=200):
    """
    Description:
        Displayes an image
    Inputs:
        path (str): File path
        dpi (int): Your monitor's pixel density
    """
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize = (width/dpi,height/dpi))
    plt.imshow(img, interpolation='nearest', aspect='auto')

df_view = df_word_count.loc[(df_word_count['word_length1']>10) & (df_word_count['word_length2']<100) & (df_word_count['word_length2']>10)] 
#df_view = df2.loc[df2['pred']==4]
print(len(df_view))
sample_n = 10
class_list = []

for idx, items in df_view.sample(n=sample_n).iterrows():
    #class_list.append(items['atg_code'])
    #print(df.iloc[idx,:])
    print(items['class'], items['word_length1'], items['word_length2'])
    display_image(img_path(items.atg_code))

# COMMAND ----------

pd.read_csv(os.path.join("file:/dbfs", "final_combine_output_untagged.csv"))

# COMMAND ----------

A = []
A.append('sss')
A

# COMMAND ----------

# ted data
df_ted = pd.read_csv(os.path.join("file:/dbfs", "final_combine_output_untagged.csv"))[['atg_code', 'pred']]
#df_ted = pd.read_csv("/dbfs/untagged_gpt_202304271343.csv")[['0', 'predicted_lables',"Class1","Class2","Class3","Class4","Class5","Class6"]]
df_ted.columns = ['atg_code', 'pred']#,"Class1","Class2","Class3","Class4","Class5","Class6"]
df = df_ted
df.groupby('pred').size()

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def display_image(path, dpi=400):
    """
    Description:
        Displayes an image
    Inputs:
        path (str): File path
        dpi (int): Your monitor's pixel density
    """
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize = (width/dpi,height/dpi))
    plt.imshow(img, interpolation='nearest', aspect='auto')

for idx, items in df.loc[df['pred']==0].sample(n=4).iterrows():
    #print(df.iloc[idx,:])
    display_image(img_path(items.atg_code))

# COMMAND ----------

