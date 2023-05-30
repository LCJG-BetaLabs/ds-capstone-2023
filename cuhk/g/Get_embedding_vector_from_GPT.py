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

code = "BWB415"
df_cleansed[df_cleansed["atg_code"] == code]

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

df_7 = df_cleansed[df_cleansed["class"] == 7]
df_7['combined_desc'] = df_7['color_desc'] + " " + df_7['prod_desc_eng'] + " " + df_7['long_desc']
df_7['cleaned_combined_desc'] = df_7['combined_desc'].apply(clean_stopwords)
df_7['cleaned_long_desc'] = df_7['long_desc'].apply(clean_stopwords)

# COMMAND ----------

gpt_model_name = 'gpt2'  # or any other GPT-2 model name
hidden_size = 768  # or any other desired hidden size  #####768 試左 55 - 65% in test data

# Instantiate the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = GPT2Model.from_pretrained(gpt_model_name, hidden_size=hidden_size)

list_embedding_df7 = []
for text in df_7['combined_desc']:
    # Tokenize the input text and convert it to a tensor
    #input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    gpt2_input = tokenizer(text, padding="max_length", max_length=100, truncation=True, return_tensors="pt")
    input_ids = gpt2_input['input_ids']
    mask = gpt2_input["attention_mask"]
    # Obtain the model's hidden states for the input tensor
    hidden_states = model(input_ids = input_ids,attention_mask = mask)[0]

    # Extract the embedding vector for the first token in the input text (which corresponds to the [CLS] token)
    embedding_vector = hidden_states[0][0]

    list_embedding_df7.append(embedding_vector.tolist())

# COMMAND ----------

embeding_vector_with_7_df = pd.DataFrame(list_embedding_df7)
final_gpt_ebd_df_with7 = pd.concat([df_7.reset_index()[["atg_code","class"]],embeding_vector_with_7_df],axis = 1, ignore_index = True)

# COMMAND ----------

#final_gpt_ebd_df_with7.to_csv('/dbfs/final_gpt_ebd_df_with7.csv')

# COMMAND ----------

df_without_7 = df_cleansed[df_cleansed["class"] != 7]
df_without_7['combined_desc'] = df_without_7['color_desc'] + " " + df_without_7['prod_desc_eng'] + " " + df_without_7['long_desc']

# COMMAND ----------

df_without_7['cleaned_combined_desc'] = df_without_7['combined_desc'].apply(clean_stopwords)
df_without_7['cleaned_long_desc'] = df_without_7['long_desc'].apply(clean_stopwords)

# COMMAND ----------

list_embedding2 = []
for text in df_without_7['combined_desc']:
    # Tokenize the input text and convert it to a tensor
    #input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    gpt2_input = tokenizer(text, padding="max_length", max_length=100, truncation=True, return_tensors="pt")
    input_ids = gpt2_input['input_ids']
    mask = gpt2_input["attention_mask"]
    # Obtain the model's hidden states for the input tensor
    hidden_states = model(input_ids = input_ids,attention_mask = mask)[0]

    # Extract the embedding vector for the first token in the input text (which corresponds to the [CLS] token)
    embedding_vector = hidden_states[0][0]

    list_embedding2.append(embedding_vector.tolist())

# COMMAND ----------

embeding_vector_without_7_df = pd.DataFrame(list_embedding2)

# COMMAND ----------

final_gpt_ebd_df_without7 = pd.concat([df_without_7.reset_index()[["atg_code","class"]],embeding_vector_without_7_df],axis = 1)

# COMMAND ----------

#final_gpt_ebd_df_without7.to_csv('/dbfs/final_gpt_ebd_df_without7.csv')

# COMMAND ----------

final_gpt_ebd_df_without7

# COMMAND ----------

df_without_7_cleaned = final_gpt_ebd_df_without7
df_without_7_cleaned["class"] = df_without_7_cleaned["class"] - 1

# COMMAND ----------

######no need to run first##############

# COMMAND ----------

gpt_model_name = 'gpt2'  # or any other GPT-2 model name
hidden_size = 768  # or any other desired hidden size  #####768 試左 55 - 65% in test data

# Instantiate the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = GPT2Model.from_pretrained(gpt_model_name, hidden_size=hidden_size)

# COMMAND ----------

list_embedding = []
for text in df_without_7_cleaned['combined_desc']:
    # Tokenize the input text and convert it to a tensor
    #input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    gpt2_input = tokenizer(text, padding="max_length", max_length=100, truncation=True, return_tensors="pt")
    input_ids = gpt2_input['input_ids']
    mask = gpt2_input["attention_mask"]
    # Obtain the model's hidden states for the input tensor
    hidden_states = model(input_ids = input_ids,attention_mask = mask)[0]

    # Extract the embedding vector for the first token in the input text (which corresponds to the [CLS] token)
    embedding_vector = hidden_states[0][0]

    list_embedding.append(embedding_vector.tolist())

# COMMAND ----------

##############################

# COMMAND ----------

df_without_7_cleaned.head()

# COMMAND ----------

X_transformed = np.array(df_without_7_cleaned.drop(['atg_code', 'class'], axis=1))

# COMMAND ----------

y = np.array(df_without_7_cleaned['class'].tolist())

# COMMAND ----------

y_transformed = K.utils.to_categorical(y)

# COMMAND ----------

#X_transformed = np.array(list_embedding)

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_transformed, y_transformed, test_size=0.2,random_state=33)

X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5,random_state=33)

# COMMAND ----------

len(y_train), len(y_val), len(y_test)

# COMMAND ----------

unique_values, counts = np.unique(np.argmax(y_train,axis = 1), return_counts=True)
# Create a bar chart
plt.bar(range(len(counts)), counts)
plt.show()
from sklearn.utils import resample

# COMMAND ----------

#Method 1 for resample

from sklearn.utils import resample

X_train_underample0 = X_train[y_train[:,0]==1]
X_train_underample1 = X_train[y_train[:,1]==1]
X_train_underample2 = X_train[y_train[:,2]==1]
X_train_underample3 = X_train[y_train[:,3]==1]
X_train_underample4 = X_train[y_train[:,4]==1]
X_train_underample5 = X_train[y_train[:,5]==1]

y_train_underample0 = y_train[y_train[:,0]==1]
y_train_underample1 = y_train[y_train[:,1]==1]
y_train_underample2 = y_train[y_train[:,2]==1]
y_train_underample3 = y_train[y_train[:,3]==1]
y_train_underample4 = y_train[y_train[:,4]==1]
y_train_underample5 = y_train[y_train[:,5]==1]

for i in range(0, 6): 
    if i != 0 and i != 3:
        exec(f"X_train_underample{i} = resample(X_train_underample{i},y_train_underample{i}, replace=True, n_samples= 100, random_state=123)")
    else:
        exec(f"X_train_underample{i} = resample(X_train_underample{i},y_train_underample{i}, replace=False, n_samples= 100, random_state=123)")

X_train_underample = np.concatenate((X_train_underample0[0], X_train_underample1[0], X_train_underample2[0], X_train_underample3[0], X_train_underample4[0], X_train_underample5[0]), axis=0)
y_train_underample = np.concatenate((X_train_underample0[1], X_train_underample1[1], X_train_underample2[1], X_train_underample3[1], X_train_underample4[1], X_train_underample5[1]), axis=0)

# COMMAND ----------

#Method 2 for resample

X_train_underample0 = X_train[y_train[:,0]==1]
X_train_underample1 = X_train[y_train[:,1]==1]
X_train_underample2 = X_train[y_train[:,2]==1]
X_train_underample3 = X_train[y_train[:,3]==1]
X_train_underample4 = X_train[y_train[:,4]==1]
X_train_underample5 = X_train[y_train[:,5]==1]

y_train_underample0 = y_train[y_train[:,0]==1]
y_train_underample1 = y_train[y_train[:,1]==1]
y_train_underample2 = y_train[y_train[:,2]==1]
y_train_underample3 = y_train[y_train[:,3]==1]
y_train_underample4 = y_train[y_train[:,4]==1]
y_train_underample5 = y_train[y_train[:,5]==1]

X_train_underample1 = resample(X_train_underample1,y_train_underample1, replace=True, n_samples= 60, random_state=123)
X_train_underample2 = resample(X_train_underample2,y_train_underample2, replace=True, n_samples= 110, random_state=123)
X_train_underample4 = resample(X_train_underample4,y_train_underample4, replace=True, n_samples= 50, random_state=123)
X_train_underample5 = resample(X_train_underample5,y_train_underample5, replace=True, n_samples= 100, random_state=123)


X_train_underample = np.concatenate((X_train_underample0, X_train_underample1[0], X_train_underample2[0], X_train_underample3, X_train_underample4[0], X_train_underample5[0]), axis=0)
y_train_underample = np.concatenate((y_train_underample0, X_train_underample1[1], X_train_underample2[1], y_train_underample3, X_train_underample4[1], X_train_underample5[1]), axis=0)

# COMMAND ----------

X_train_underample.shape

# COMMAND ----------

unique_values, counts = np.unique(np.argmax(y_train_underample,axis = 1), return_counts=True)

# Create a bar chart
plt.bar(range(len(counts)), counts)
plt.show()
from sklearn.utils import resample

# COMMAND ----------

unique_values, counts = np.unique(np.argmax(y_val,axis = 1), return_counts=True)
# Create a bar chart
plt.bar(range(len(counts)), counts)
plt.show()
from sklearn.utils import resample

# COMMAND ----------

X_val_underample0 = X_val[y_val[:,0]==1]
X_val_underample1 = X_val[y_val[:,1]==1]
X_val_underample2 = X_val[y_val[:,2]==1]
X_val_underample3 = X_val[y_val[:,3]==1]
X_val_underample4 = X_val[y_val[:,4]==1]
X_val_underample5 = X_val[y_val[:,5]==1]

y_val_underample0 = y_val[y_val[:,0]==1]
y_val_underample1 = y_val[y_val[:,1]==1]
y_val_underample2 = y_val[y_val[:,2]==1]
y_val_underample3 = y_val[y_val[:,3]==1]
y_val_underample4 = y_val[y_val[:,4]==1]
y_val_underample5 = y_val[y_val[:,5]==1]

for i in range(0, 6): 
    if i != 0 and i != 3:
        exec(f"X_val_underample{i} = resample(X_val_underample{i},y_val_underample{i}, replace=True, n_samples= 10, random_state=123)")
    else:
        exec(f"X_val_underample{i} = resample(X_val_underample{i},y_val_underample{i}, replace=False, n_samples= 10, random_state=123)")

X_val_underample = np.concatenate((X_val_underample0[0], X_val_underample1[0], X_val_underample2[0], X_val_underample3[0], X_val_underample4[0], X_val_underample5[0]), axis=0)
y_val_underample = np.concatenate((X_val_underample0[1], X_val_underample1[1], X_val_underample2[1], X_val_underample3[1], X_val_underample4[1], X_val_underample5[1]), axis=0)

# COMMAND ----------

unique_values, counts = np.unique(np.argmax(y_val_underample,axis = 1), return_counts=True)
# Create a bar chart
plt.bar(range(len(counts)), counts)
plt.show()
from sklearn.utils import resample

# COMMAND ----------

from sklearn.metrics import accuracy_score
from keras.layers import Flatten
EPOCH_SIZE = 200

FILEPATH = "/dbfs/text_model_v1"

# Early stopping  
K.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_SIZE/10),
check_point = K.callbacks.ModelCheckpoint(filepath=FILEPATH, monitor="val_loss", mode="min", save_best_only=True,)

# Define neural network architecture
input_shape = (X_train.shape[1],)
num_classes = y_train.shape[1]
model = K.models.Sequential()
model.add(K.layers.Dense(1024, input_shape = input_shape, activation="relu"))
model.add(K.layers.Dropout(0.2))

#model.add(K.layers.BatchNormalization())
#model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train_underample, y_train_underample, batch_size=8, epochs=200, verbose=1, validation_data=(X_val, y_val),callbacks=[check_point])

model.summary()
#model.save(FILEPATH)


# Evaluate the model on test data
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
print(list(zip(y,y_pred_class)))


# COMMAND ----------

class_mapping = {
    1: 'graphic_print',
    2: 'multi_color',
    3: 'word_print',
    4: 'plain',
    5: 'checks',
    6: 'stripe',
    7: 'untagged'
}

def img_path(atg_code):
    return f"/dbfs/image/{atg_code}_in_xl.jpg"

def load_image(atg_code, resized_fac = 0.2):
    img     = cv2.imread(img_path(atg_code))
    h, _, _ = img.shape
    img   = img[0:int(h*2/3), :, :] # preserve upper half of the image only
    h, w, _ = img.shape
    resized = cv2.resize(img, (int(w*resized_fac), int(h*resized_fac)), interpolation = cv2.INTER_AREA)
    return resized

# show image and how we cut the image in half
plt.figure()
f, axarr = plt.subplots(1,6, figsize=(25, 25)) 
#for i, (idx, items) in zip(range(2), df_class7.loc[df_class7['pred']==6].sample(n=6).iterrows()):
#    print(class_mapping[items.pred])
#    axarr[i].imshow(cv2.cvtColor(load_image(items.atg_code), cv2.COLOR_BGR2RGB))

class_num = 1
print(class_mapping[class_num])
for i, (idx, items) in zip(range(6), df_cleansed.loc[df_cleansed['class']==class_num].sample(n=6).iterrows()):
    print(items.color_desc)
    axarr[i].imshow(cv2.cvtColor(load_image(items.atg_code), cv2.COLOR_BGR2RGB))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

test_predicted_prob = model.predict(X_test)

# COMMAND ----------

predicted_labels = np.argmax(test_predicted_prob, axis=1)

# COMMAND ----------

predicted_labels

# COMMAND ----------

true_labels = np.argmax(y_test, axis = 1)

# COMMAND ----------

len(y_test)

# COMMAND ----------

sum(predicted_labels == true_labels) / len(true_labels)

# COMMAND ----------

# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------

X_untagged = np.array(final_gpt_ebd_df_with7.drop([0, 1], axis=1))

# COMMAND ----------

X_untagged

# COMMAND ----------

untag_predicted_prob = model.predict(X_untagged)
predicted_labels = np.argmax(untag_predicted_prob, axis=1)

# COMMAND ----------

pred_untag = final_gpt_ebd_df_with7[[0,1]]

# COMMAND ----------

pred_untag["predicted_lables"] = predicted_labels

# COMMAND ----------

pred_untag.to_csv('/dbfs/gpt_untagged_result.csv')

# COMMAND ----------

pred_untag

# COMMAND ----------

# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

#model.save("/dbfs/ted_model_v1")

# COMMAND ----------

np.random.seed(112)
df_train, df_val, df_test = np.split(df_without_7_cleaned.sample(frac=1, random_state=35),[int(0.8*len(df_without_7_cleaned)), int(0.9*len(df_without_7_cleaned))])

print(len(df_train), len(df_val), len(df_test))

# COMMAND ----------

from sklearn.utils import resample

df_train_underample0 = df_train.loc[df_train['class']==0]
df_train_underample1 = df_train.loc[df_train['class']==1]
df_train_underample2 = df_train.loc[df_train['class']==2]
df_train_underample3 = df_train.loc[df_train['class']==3]
df_train_underample4 = df_train.loc[df_train['class']==4]
df_train_underample5 = df_train.loc[df_train['class']==5]

for i in range(0, 6): 
    if i != 0 and i != 3:
        exec(f"df_train_underample{i} = resample(df_train_underample{i}, replace=True, n_samples= 100, random_state=123)")
    else:
        exec(f"df_train_underample{i} = resample(df_train_underample{i}, replace=False, n_samples= 100, random_state=123)")

df_train = pd.concat((df_train_underample0,df_train_underample1, df_train_underample2, df_train_underample3, df_train_underample4, df_train_underample5))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

1213

# COMMAND ----------

gpt_model_name = 'gpt2'  # or any other GPT-2 model name
hidden_size = 768  # or any other desired hidden size

# Instantiate the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
model = GPT2Model.from_pretrained(gpt_model_name, hidden_size=hidden_size)

# Define an input text
input_text = "Hello, World!"

# Tokenize the input text and convert it to a tensor
input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)

# Obtain the model's hidden states for the input tensor
hidden_states = model(input_ids)[0]

# Extract the embedding vector for the first token in the input text (which corresponds to the [CLS] token)
embedding_vector = hidden_states[0][0]

# Print the embedding vector
print(embedding_vector)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

