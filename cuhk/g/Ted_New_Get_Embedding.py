# Databricks notebook source
import pandas as pd
import os
df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow.keras as K 

# COMMAND ----------

gpt_model_name = 'gpt2'  # or any other GPT-2 model name
hidden_size = 768  # or any other desired hidden size  #####768 試左 55 - 65% in test data

# Instantiate the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = GPT2Model.from_pretrained(gpt_model_name, hidden_size=hidden_size)

# COMMAND ----------

# Clean the text by removing stopwords and punctuation
import nltk
nltk.download('stopwords') # Download the stopwords data
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import string

stop_words = set(stopwords.words('english'))

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

df_cleansed['combined_desc'] = df_cleansed['color_desc'] + " " + df_cleansed['prod_desc_eng'] + " " + df_cleansed['long_desc']
df_cleansed['cleaned_combined_desc'] = df_cleansed['combined_desc'].apply(clean_stopwords)
df_cleansed['cleaned_long_desc'] = df_cleansed['long_desc'].apply(clean_stopwords)

# COMMAND ----------

list_embedding_df = []
for text in df_cleansed['combined_desc']:
    # Tokenize the input text and convert it to a tensor
    #input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    gpt2_input = tokenizer(text, padding="max_length", max_length=125, truncation=True, return_tensors="pt")
    input_ids = gpt2_input['input_ids']
    mask = gpt2_input["attention_mask"]
    # Obtain the model's hidden states for the input tensor
    hidden_states = model(input_ids = input_ids,attention_mask = mask)[0]

    # Extract the embedding vector for the first token in the input text (which corresponds to the [CLS] token)
    embedding_vector = hidden_states[0][0]

    list_embedding_df.append(embedding_vector.tolist())

# COMMAND ----------

embeding_vector_df = pd.DataFrame(list_embedding_df)

# COMMAND ----------

final_output_100 = pd.concat([df_cleansed.reset_index()[["atg_code","class"]],embeding_vector_df],axis = 1, ignore_index = True)

# COMMAND ----------

final_output_100.to_csv('/dbfs/final_gpt_ebd_df_150_04271345.csv',index = False)

# COMMAND ----------

#final_gpt_ebd_df_150_04271345

# COMMAND ----------

final_output_100

# COMMAND ----------

df_cleansed

# COMMAND ----------

df_cleansed.groupby('atg_subclass_desc').count().plot.bar()

# COMMAND ----------

table = pd.crosstab(df_cleansed['atg_subclass_desc'], df_cleansed['class'])

# Plot the contingency table using a stacked bar plot
table.plot(kind='bar', stacked=True)

# Add labels and a legend to the plot
plt.title('Color-Shape Combinations')
plt.xlabel('Color')
plt.ylabel('Count')

# COMMAND ----------

##################Test the longest word size###############

# COMMAND ----------

def count_words(text):
    return len(text.split())

# COMMAND ----------

columns_name = 'long_desc'

# COMMAND ----------

df_cleansed['word_count'] = df_cleansed[columns_name].apply(count_words)

# COMMAND ----------

df_cleansed[columns_name].iloc[np.argmax(df_cleansed['word_count'])]

# COMMAND ----------

max(df_cleansed['word_count'])

# COMMAND ----------

gpt2_input = tokenizer(df_cleansed[columns_name].iloc[np.argmax(df_cleansed['word_count'])], padding="max_length", max_length=101, truncation=True, return_tensors="pt")
input_ids = gpt2_input['input_ids']
mask = gpt2_input["attention_mask"]

# COMMAND ----------

mask

# COMMAND ----------

for text in df_cleansed[columns_name]:
    print(text)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

#care max 43 [20,43,60]
#prod_desc_eng max 43 [,43,]
#color = 4
#long desc max 100

max_length = 50

list_embedding_df = []
for text in df_cleansed[columns_name]:
    # Tokenize the input text and convert it to a tensor
    #input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    gpt2_input = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    input_ids = gpt2_input['input_ids']
    mask = gpt2_input["attention_mask"]
    # Obtain the model's hidden states for the input tensor
    hidden_states = model(input_ids = input_ids,attention_mask = mask)[0]

    # Extract the embedding vector for the first token in the input text (which corresponds to the [CLS] token)
    embedding_vector = hidden_states[0][0]

    list_embedding_df.append(embedding_vector.tolist())

# COMMAND ----------

embeding_vector_df = pd.DataFrame(list_embedding_df)
final_output_100 = pd.concat([df_cleansed.reset_index()[["atg_code","class"]],embeding_vector_df],axis = 1, ignore_index = True)
final_output_100.to_csv(f'/dbfs/final_gpt_ebd_df_{columns_name}_{max_length}.csv',index = False)


# COMMAND ----------

pd.read_csv(f'/dbfs/final_gpt_ebd_df_{columns_name}_{max_length}.csv')

# COMMAND ----------

pd.read_csv(f'/dbfs/final_gpt_ebd_df_{columns_name}_{max_length}.csv')

# COMMAND ----------

os.listdir('/dbfs/')

# COMMAND ----------

