# Databricks notebook source
pip install opencv-python

# COMMAND ----------

class_mapping = {
    1: 'graphic_print',
    2: 'multi_color',
    3: 'word_print',
    4: 'plain',
    5: 'checks',
    6: 'stripe',
    7: 'untagged',
    8: 'additional checks',
    9: 'additional diamonds'
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


# Clean the text by removing stopwords and punctuation
nltk.download('stopwords') # Download the stopwords data
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


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

def clean_stopwords(text):
    # Define a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    # Use the translation table to remove punctuation from the text
    text = text.translate(translator)

    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if not word in stop_words]

    filtered_text = ' '.join(filtered_tokens) 
    return filtered_text  


# COMMAND ----------

# DBTITLE 1,Text Preprocessing
df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
#df_cleansed['cleaned_long_desc'] = df_cleansed['long_desc'].apply(clean_stopwords)
#df_cleansed['cleaned_color_desc'] = df_cleansed['color_desc'].apply(clean_stopwords)
#df_cleansed['cleaned_brand_desc'] = df_cleansed['brand_desc'].apply(clean_stopwords)
#df_cleansed['cleaned_prod_desc_eng'] = df_cleansed['prod_desc_eng'].apply(clean_stopwords)

df_cleansed['conbined_desc'] = df_cleansed['long_desc'] + " " + df_cleansed['color_desc'] + " " + df_cleansed['prod_desc_eng'] # +  " " + df_cleansed['brand_desc'] 

df_cleansed['cleaned_conbined_desc'] = df_cleansed['conbined_desc'].apply(clean_stopwords)

# COMMAND ----------

df_cleansed['cleaned_conbined_desc']

# COMMAND ----------

y = np.array(df_cleansed[df_cleansed['class']!=7]['class'].tolist())
X = df_cleansed[df_cleansed['class']!=7]['cleaned_conbined_desc'].tolist()
len(X), len(y)

# Convert extracted keywords into feature vectors 
vectorizer = TfidfVectorizer(stop_words='english',min_df=0.1,max_df=0.5, sublinear_tf=True, max_features=41) # could be changed to CountVectorizer
X_transformed_text = vectorizer.fit_transform(X).toarray() # long desc

#vectorizer_prod = TfidfVectorizer(stop_words='english',min_df=0.1,max_df=0.5, sublinear_tf=True, max_features=400)
#X_transformed_text_prod = vectorizer_prod.fit_transform(df_cleansed[df_cleansed['class']!=7]['cleaned_prod_desc_eng'].tolist()).toarray() # prod desc

#vectorizer_word = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=7) 
#X_transformed_text_color = vectorizer_word.fit_transform(df_cleansed[df_cleansed['class']!=7]['cleaned_color_desc'].tolist()).toarray() # long desc
#X_transformed_text_brand = vectorizer_word.fit_transform(df_cleansed[df_cleansed['class']!=7]['cleaned_brand_desc'].tolist()).toarray() # long desc

print(X_transformed_text.shape)

# Convert class labels to numerical values
y_transformed = K.utils.to_categorical(y)
print(y_transformed.shape)

# COMMAND ----------

# convert numpy array of text embedding to pandas dataFrame
# adding prefix to col name to avoid duplicate col name in later process
df_embs_text = pd.DataFrame(X_transformed_text, columns=[f"text{i}" for i in range(X_transformed_text.shape[1])])
#df_embs_text_color = pd.DataFrame(X_transformed_text_color, columns=[f"color{i}" for i in range(X_transformed_text_color.shape[1])])
#df_embs_text_brand = pd.DataFrame(X_transformed_text_brand, columns=[f"brand{i}" for i in range(X_transformed_text_brand.shape[1])])
#df_embs_text_prod = pd.DataFrame(X_transformed_text_prod, columns=[f"brand{i}" for i in range(X_transformed_text_prod.shape[1])])

# COMMAND ----------



# COMMAND ----------

df_embs_gpt = pd.read_csv(os.path.join("file:/dbfs", "final_gpt_ebd_df_without7.csv"))
df_embs_gpt

# COMMAND ----------

df_embs_gpt.groupby('atg_code').size()

# COMMAND ----------

# DBTITLE 1,Image processing - import and cleansing embedding
df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new.csv"))
df_embs_original = pd.concat([df_embs_original, df_cleansed['atg_code']], axis=1) 

df_embs = df_embs_original.loc[(df_embs_original['Y']!=7)] # remove untagged items
df_embs = df_embs.reset_index(drop=True)
df_embs = df_embs.iloc[:,1:]
df_embs

# normalize data
#df_embs_numpy = minmax_scale(df_embs.iloc[:,1:-2], feature_range=(0, 1))
#df_embs = pd.concat([pd.DataFrame(df_embs_numpy), df_embs.iloc[:,-2:]], axis=1, ignore_index=True)
#df_embs.head()

# COMMAND ----------

df_embs.groupby('Y').size()

# COMMAND ----------

# combining text and image embeddings
df_embs = pd.concat([
    df_embs.iloc[:, :-2], # image embedding
    df_embs_text, # text embedding - long desc
    #df_embs_text_color, # test embedding - color desc
    #df_embs_text_brand, # test embedding - brand desc
    #df_embs_text_prod, # test embedding - prod desc
    df_embs.iloc[:, -2:] # Y & atg_code
    ], axis=1)

df_embs = df_embs.rename(columns={2048:"Y", 2049:"atg_code"})
df_embs

# COMMAND ----------

# check if y of image data equal to y of text data
(df_embs.Y.to_numpy()==y).all()

# COMMAND ----------

# #train test split by class
# df_embs_train_undersample = pd.DataFrame()
# df_embs_test = pd.DataFrame()
# for i in range(1, 8):
#    msk = np.random.rand(len(df_embs.loc[df_embs['Y']==i])) < 0.8
#    df_embs_train_undersample = df_embs_train_undersample.append(df_embs.loc[df_embs['Y']==i][msk])
#    df_embs_test = df_embs_test.append(df_embs.loc[df_embs['Y']==i][~msk])

# df_embs_train_undersample.shape, df_embs_test.shape

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train = pd.DataFrame()
X_test = pd.DataFrame()
X_val = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
y_val = pd.DataFrame()

for i in df_embs['Y'].unique():
    df_embs_single_class = df_embs.loc[df_embs['Y']==4]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df_embs_single_class.iloc[:,:-2], df_embs_single_class["Y"], test_size=0.2,random_state=33)
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train1, y_train1, test_size=0.2, random_state=42)

    X_train = X_train.append(X_train1.reset_index(drop=True))
    X_val = X_val.append(X_val1.reset_index(drop=True))
    X_test = X_test.append(X_test1.reset_index(drop=True))
    y_train = y_train.append(y_train1.reset_index(drop=True))
    y_test = y_test.append(y_test1.reset_index(drop=True))
    y_val = y_val.append(y_val1.reset_index(drop=True))

# COMMAND ----------

# resample


# COMMAND ----------

# DBTITLE 1,Resample for unbalance class
# from sklearn.utils import resample

# df_embs_underample4 = df_embs_train_undersample.loc[df_embs_train_undersample['Y']==4]
# df_embs_underample1 = df_embs_train_undersample.loc[df_embs_train_undersample['Y']==1] # not resmapled
# df_embs_underample2 = df_embs_train_undersample.loc[df_embs_train_undersample['Y']==2]
# df_embs_underample3 = df_embs_train_undersample.loc[df_embs_train_undersample['Y']==3]
# df_embs_underample5 = df_embs_train_undersample.loc[df_embs_train_undersample['Y']==5]
# df_embs_underample6 = df_embs_train_undersample.loc[df_embs_train_undersample['Y']==6]
# df_embs_underample8 = df_embs_train_undersample.loc[(df_embs_train_undersample['Y']==8) | (df_embs_train_undersample['Y']==9)]
# df_embs_underample9 = df_embs_train_undersample.loc[df_embs_train_undersample['Y']==9]

# sample_size = 50

# for i in [4, 1, 2, 3, 6, 5]: # group 1 is not resampled
#    if df_embs_train_undersample.loc[df_embs_train_undersample['Y']==i].shape[0]>sample_size:
#        exec(f"df_embs_resample{i} = df_embs_underample{i}.sample(n={sample_size})")
#    else:
#        exec(f"df_embs_resample{i} = resample(df_embs_underample{i}, replace=True, n_samples={sample_size}, random_state=123)")

# df_embs_train = pd.concat((df_embs_resample1, df_embs_resample2, df_embs_resample3, df_embs_resample4, df_embs_resample5, df_embs_resample6))

# COMMAND ----------

print(f"class count (BEFORE train test split): ")
print(df_embs.groupby('Y').size())

# COMMAND ----------

print(f"class count (BEFORE train test split): ")
print(df_embs.groupby('Y').size())
print(f"\nclass count (AFTER train test split): ")
print(df_embs_train_undersample.groupby('Y').size())
print(f"\nclass count (AFTER resampling): ")
print(df_embs_train.groupby('Y').size())
print(f"\nunique class count (AFTER resampling): ")
print(df_embs_train.drop_duplicates().groupby('Y').size())

# COMMAND ----------

#X_transformed_image = df_embs.iloc[:,1:-2].to_numpy()

#X_transformed = np.concatenate((X_transformed_text, X_transformed_image), axis=1)
#X_transformed_text.shape, X_transformed_image.shape, X_transformed.shape

# COMMAND ----------

# testing only
X_transformed = df_embs_train.iloc[:,:-2].to_numpy()
y_transformed = K.utils.to_categorical(df_embs_train.Y)

X_transformed.shape, y_transformed.shape

# COMMAND ----------

df_embs_test.groupby('Y').size()

# COMMAND ----------

#from sklearn.model_selection import train_test_split

# Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2,random_state=33)
#X_train.shape, y_train.shape, X_test.shape, y_test.shape

# COMMAND ----------

df_embs_test.groupby('Y').size()

# COMMAND ----------

X_train = X_transformed
y_train = y_transformed
X_test = df_embs_test.iloc[:,:-2].to_numpy()
y_test = K.utils.to_categorical(df_embs_test.Y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# COMMAND ----------

#normalize data
from sklearn import preprocessing

normalizer = preprocessing.Normalizer()
normalized_train_X = normalizer.fit_transform(X_train)
print(normalized_train_X.shape)

normalized_test_X = normalizer.transform(X_test)
normalized_test_X.shape

# COMMAND ----------

from sklearn.metrics import accuracy_score
from sklearn import metrics

FILEPATH = "/dbfs/combined_model_v1"

EPOCH_SIZE = 160

# Early stopping  
check_point = [
    K.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_SIZE/10),
    K.callbacks.ModelCheckpoint(filepath=FILEPATH, monitor="val_loss", mode="min", save_best_only=True,)
]

# Define neural network architecture
input_shape = (normalized_train_X.shape[1],)
num_classes = y_train.shape[1]

model = K.models.Sequential()
model.add(K.layers.Dense(64, input_shape = input_shape, activation="relu"))
#model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(normalized_train_X, y_train, batch_size=8, epochs=EPOCH_SIZE, verbose=1, validation_data=(normalized_test_X, y_test),callbacks=[check_point])
print(len(history.history['val_loss']))

model.summary()
model.save(FILEPATH)


# Evaluate the model on test data
y_pred = model.predict(normalized_test_X)
y_pred_class = np.argmax(y_pred, axis=1)


# consusion matrix
confusion_matrix = metrics.confusion_matrix(df_embs_test.Y, y_pred_class)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

# COMMAND ----------

normalized_test_X.shape

# COMMAND ----------

df_embs_class7 = df_embs_original.loc[df_embs_original['Y']==7] # remove untagged items
df_embs_class7 = df_embs_class7.reset_index(drop=True)
print(df_embs_class7.shape)

# normalize data
df_embs_numpy_class7 = minmax_scale(df_embs_class7.iloc[:,1:-2], feature_range=(0, 1))
df_embs_class7 = pd.concat([pd.DataFrame(df_embs_numpy_class7), df_embs_class7.iloc[:,-2:]], axis=1, ignore_index=True)
print(df_embs_class7.shape)

# text data
X7 = df_cleansed[df_cleansed['class']==7]['cleaned_conbined_desc'].tolist()
X_transformed_text_class7 = vectorizer.fit_transform(X7).toarray()
print(X_transformed_text_class7.shape)

df_embs_text = pd.DataFrame(X_transformed_text_class7, columns=[f"text{i}" for i in range(X_transformed_text_class7.shape[1])])
print(df_embs_text.shape)

df_embs_class7 = pd.concat([df_embs_class7.iloc[:, :-2] , df_embs_text, df_embs_class7.iloc[:, -2:]], axis=1) #df_embs_text
df_embs_class7 = df_embs_class7.rename(columns={2048:"Y", 2049:"atg_code"})
df_embs_class7
print(df_embs_class7.shape)

# COMMAND ----------

df_class7

# COMMAND ----------

X_class7 = df_embs_class7.iloc[:, :-2].to_numpy()

y_pred_class7 = model.predict(X_class7)
y_pred_class7 = np.argmax(y_pred_class7, axis=1)

df_class7 = df_embs_original.loc[df_embs_original["Y"]==7][['Y', 'atg_code']]

df_class7['pred'] = y_pred_class7
df_class7

# COMMAND ----------

# show image and how we cut the image in half
plt.figure()
f, axarr = plt.subplots(1,3, figsize=(25, 25)) 
for i, (idx, items) in zip(range(3), df_class7.loc[df_class7['Y']==7].sample(n=3).iterrows()):
    print(class_mapping[items.pred])
    axarr[i].imshow(cv2.cvtColor(load_image(items.atg_code), cv2.COLOR_BGR2RGB))

#class_num = 1
#print(class_mapping[class_num])
#for i, (idx, items) in zip(range(6), df_cleansed.loc[df_cleansed['class']==class_num].sample(n=6).iterrows()):
#    print(items.color_desc)
#    axarr[i].imshow(cv2.cvtColor(load_image(items.atg_code), cv2.COLOR_BGR2RGB))



# COMMAND ----------

from sklearn.metrics import accuracy_score
from sklearn import metrics

FILEPATH = "/dbfs/combined_model_v1"

EPOCH_SIZE = 30

# Early stopping  
check_point = [
    K.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_SIZE/10),
    K.callbacks.ModelCheckpoint(filepath=FILEPATH, monitor="val_loss", mode="min", save_best_only=True,)
]

# Define neural network architecture
input_shape = (X_train.shape[1],)
num_classes = y_train.shape[1]

model = K.models.Sequential()
model.add(K.layers.Dense(128, input_shape = input_shape, activation="relu"))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=EPOCH_SIZE, verbose=1, validation_data=(X_test, y_test),callbacks=[check_point])
print(len(history.history['val_loss']))

model.summary()
model.save(FILEPATH)

#best val accuracy: 72.17%

# Evaluate the model on test data
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)


# consusion matrix
confusion_matrix = metrics.confusion_matrix(df_embs_test.Y, y_pred_class)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

# COMMAND ----------

from sklearn.metrics import accuracy_score

FILEPATH = "/dbfs/text_model_v4"

# Early stopping  
check_point = K.callbacks.ModelCheckpoint(filepath=FILEPATH,
                                              monitor="val_loss",
                                              mode="min",
                                              save_best_only=True,
                                              )

# Define neural network architecture
input_shape = (X_train.shape[1],)
num_classes = y_train.shape[1]

model = K.models.Sequential()
model.add(K.layers.Dense(32, input_shape = input_shape, activation="relu"))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, batch_size=16, epochs=25, verbose=1, validation_data=(X_test, y_test),callbacks=[check_point])
print(len(history.history['val_loss']))

model.summary()
model.save(FILEPATH)

#best val accuracy: 72.17%

# Evaluate the model on test data
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
print(list(zip(y,y_pred_class)))

# COMMAND ----------

# DBTITLE 1,Use fine-tuned GPT2
import io
import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

# Set seed for reproducibility.
set_seed(123)

# Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
epochs = 4

# Number of batches - depending on the max sequence length and GPU memory.
# For 512 sequence length batch of 10 works without cuda memory issues.
# For small sequence length can try batch of 32 or higher.
batch_size = 32

# Pad or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = 60

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Name of transformers model - will use already pretrained model.
# Path of transformer model - will load your own model from local disk.
model_name_or_path = 'gpt2'

# Dictionary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids = {'neg': 0, 'pos': 1}

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

# COMMAND ----------

class MovieReviewsDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens.

  This class is built with reusability in mind: it can be used as is as.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.

  """

  def __init__(self, path, use_tokenizer):

    # Check if path exists.
    if not os.path.isdir(path):
      # Raise error if path is invalid.
      raise ValueError('Invalid `path` variable! Needs to be a directory')
    self.texts = []
    self.labels = []
    # Since the labels are defined by folders with data we loop 
    # through each label.
    for label in ['pos', 'neg']:
      sentiment_path = os.path.join(path, label)

      # Get all files from path.
      files_names = os.listdir(sentiment_path)#[:10] # Sample for debugging.
      # Go through each file and read its content.
      for file_name in tqdm(files_names, desc=f'{label} files'):
        file_path = os.path.join(sentiment_path, file_name)

        # Read content.
        content = io.open(file_path, mode='r', encoding='utf-8').read()
        # Fix any unicode issues.
        content = fix_text(content)
        # Save content.
        self.texts.append(content)
        # Save encode labels.
        self.labels.append(label)

    # Number of exmaples.
    self.n_examples = len(self.labels)
    

    return

  def __len__(self):
    r"""When used `len` return the number of examples.

    """
    
    return self.n_examples

  def __getitem__(self, item):
    r"""Given an index return an example from the position.
    
    Arguments:

      item (:obj:`int`):
          Index position to pick an example to return.

    Returns:
      :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
      asociated labels.

    """

    return {'text':self.texts[item],
            'label':self.labels[item]}



class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs


def train(dataloader, optimizer_, scheduler_, device_):
  r"""
  Train pytorch model on a single pass through the data loader.

  It will use the global variable `model` which is the transformer model 
  loaded on `_device` that we want to train on.

  This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

  Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

      optimizer_ (:obj:`transformers.optimization.AdamW`):
          Optimizer used for training.

      scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
          PyTorch scheduler.

      device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.

  Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss].
  """

  # Use global variable for model.
  global model

  # Tracking variables.
  predictions_labels = []
  true_labels = []
  # Total loss for this epoch.
  total_loss = 0

  # Put the model into training mode.
  model.train()

  # For each batch of training data...
  for batch in tqdm(dataloader, total=len(dataloader)):

    # Add original labels - use later for evaluation.
    true_labels += batch['labels'].numpy().flatten().tolist()
    
    # move batch to device
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
    
    # Always clear any previously calculated gradients before performing a
    # backward pass.
    model.zero_grad()

    # Perform a forward pass (evaluate the model on this training batch).
    # This will return the loss (rather than the model output) because we
    # have provided the `labels`.
    # The documentation for this a bert model function is here: 
    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
    outputs = model(**batch)

    # The call to `model` always returns a tuple, so we need to pull the 
    # loss value out of the tuple along with the logits. We will use logits
    # later to calculate training accuracy.
    loss, logits = outputs[:2]

    # Accumulate the training loss over all of the batches so that we can
    # calculate the average loss at the end. `loss` is a Tensor containing a
    # single value; the `.item()` function just returns the Python value 
    # from the tensor.
    total_loss += loss.item()

    # Perform a backward pass to calculate the gradients.
    loss.backward()

    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update parameters and take a step using the computed gradient.
    # The optimizer dictates the "update rule"--how the parameters are
    # modified based on their gradients, the learning rate, etc.
    optimizer.step()

    # Update the learning rate.
    scheduler.step()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Convert these logits to list of predicted labels values.
    predictions_labels += logits.argmax(axis=-1).flatten().tolist()

  # Calculate the average loss over the training data.
  avg_epoch_loss = total_loss / len(dataloader)
  
  # Return all true labels and prediction for future evaluations.
  return true_labels, predictions_labels, avg_epoch_loss



def validation(dataloader, device_):
  r"""Validation function to evaluate model performance on a 
  separate set of data.

  This function will return the true and predicted labels so we can use later
  to evaluate the model's performance.

  This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

  Arguments:

    dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

    device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.

  Returns:
    
    :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss]
  """

  # Use global variable for model.
  global model

  # Tracking variables
  predictions_labels = []
  true_labels = []
  #total loss for this epoch.
  total_loss = 0

  # Put the model in evaluation mode--the dropout layers behave differently
  # during evaluation.
  model.eval()

  # Evaluate data for one epoch
  for batch in tqdm(dataloader, total=len(dataloader)):

    # add original labels
    true_labels += batch['labels'].numpy().flatten().tolist()

    # move batch to device
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up validation
    with torch.no_grad():        

        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        # token_type_ids is the same as the "segment ids", which 
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to to calculate training accuracy.
        loss, logits = outputs[:2]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()
        
        # get predicitons to list
        predict_content = logits.argmax(axis=-1).flatten().tolist()

        # update list
        predictions_labels += predict_content

  # Calculate the average loss over the training data.
  avg_epoch_loss = total_loss / len(dataloader)

  # Return all true labels and prediciton for future evaluations.
  return true_labels, predictions_labels, avg_epoch_loss


# COMMAND ----------

# Get model configuration.
print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# default to left padding
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token


# Get the actual model.
print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)


# COMMAND ----------

# Create data collator to encode text and labels into numbers.
gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)


print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = MovieReviewsDataset(path='/content/aclImdb/train', 
                               use_tokenizer=tokenizer)
print('Created `train_dataset` with %d examples!'%len(train_dataset))

# Move pytorch dataset into dataloader.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print()

print('Dealing with Validation...')
# Create pytorch dataset.
valid_dataset =  MovieReviewsDataset(path='/content/aclImdb/test', 
                               use_tokenizer=tokenizer)
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))


# COMMAND ----------

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )

# Total number of training steps is number of batches * number of epochs.
# `train_dataloader` contains batched data so `len(train_dataloader)` gives 
# us the number of batches.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# Store the average loss after each epoch so we can plot them.
all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}

# Loop through each epoch.
print('Epoch')
for epoch in tqdm(range(epochs)):
  print()
  print('Training on batches...')
  # Perform one full pass over the training set.
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)

  # Get prediction form model on validation data. 
  print('Validation on batches...')
  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)

  # Print loss and accuracy values to see how training evolves.
  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()

  # Store the loss value for plotting the learning curve.
  all_loss['train_loss'].append(train_loss)
  all_loss['val_loss'].append(val_loss)
  all_acc['train_acc'].append(train_acc)
  all_acc['val_acc'].append(val_acc)

# Plot loss curves.
plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

# Plot accuracy curves.
plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
