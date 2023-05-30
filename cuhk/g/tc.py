# Databricks notebook source
pip install datasets transformers[sentencepiece]

# COMMAND ----------



# COMMAND ----------

import pandas as pd

new_tag = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_cuhk_team_g@ijam.onmicrosoft.com/new_tag_2.csv")
new_tag = ds_true.select("*").toPandas()
new_tag

# COMMAND ----------

df1 = pd.read_csv("/dbfs/FileStore/shared_uploads/capstone2023_cuhk_team_g@ijam.onmicrosoft.com/new_tag2a.csv")
df1

# COMMAND ----------



# COMMAND ----------

import pandas as pd

df1 = pd.read_csv("/dbfs/FileStore/shared_uploads/capstone2023_cuhk_team_g@ijam.onmicrosoft.com/new_tag2.csv")\
df1

# COMMAND ----------

import pandas as pd

ds_true = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_cuhk_team_g@ijam.onmicrosoft.com/True-1.csv")
ds_true = ds_true.select("*").toPandas()
ds_true["class"] = 1



ds_fake = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_cuhk_team_g@ijam.onmicrosoft.com/Fake.csv")
ds_fake = ds_fake.select("*").toPandas()
ds_fake["class"] = 0

ds_complete =  pd.concat([ds_true, ds_fake], axis=0)
ds_complete.head()


# COMMAND ----------

ds_complete["class"].value_counts()

# COMMAND ----------

from datasets import Dataset, DatasetDict
dataset = Dataset.from_pandas(ds_complete)
dataset = dataset.remove_columns(['subject', 'date', '__index_level_0__'])
dataset = dataset.shuffle(seed = 555)
dataset

# COMMAND ----------

train_testvalid = dataset.train_test_split(test_size=0.4)

test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

dataset

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

model_name = "ydshieh/tiny-random-gptj-for-sequence-classification"

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# COMMAND ----------

# def tokenize_function(examples):
#   return tokenizer(examples["text"], truncation = True, max_length = 512, padding = True)

# tokenized_datasets = dataset.map(tokenize_function, batched=True)


def tokenize_function(examples):
  return tokenizer(examples["text"], truncation = True, max_length = 512, padding = True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
#tokenized_datasets = dataset.map(tokenize_function)#, batched=True)

# COMMAND ----------

dataset

# COMMAND ----------

tokenized_datasets

# COMMAND ----------

tokenized_datasets = tokenized_datasets.rename_column("class", "labels")

# COMMAND ----------

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# COMMAND ----------

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis = -1)
    return metric.compute(predictions=predictions, references=labels)

# COMMAND ----------

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  fp16 = True,
                                  per_device_train_batch_size= 64,
                                  per_device_eval_batch_size= 64
                                  )

# COMMAND ----------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# COMMAND ----------

trainer.train()

# COMMAND ----------

predictions = trainer.predict(tokenized_datasets['test'])

# COMMAND ----------

predictions[2]

# COMMAND ----------

from sklearn.metrics import accuracy_score
print(accuracy_score(np.argmax(predictions[0][0], axis=-1), tokenized_datasets['test']['labels']))

# COMMAND ----------

pip install opencv-python

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

# COMMAND ----------



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

img_array = load_image("BWA955")
plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
print(img_array.shape)


# COMMAND ----------

import pandas as pd
import os

pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))