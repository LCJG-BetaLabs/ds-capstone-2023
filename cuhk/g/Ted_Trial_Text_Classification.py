# Databricks notebook source
import pandas as pd
import os
df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))

# COMMAND ----------

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# COMMAND ----------

df_without_7 = df_cleansed[df_cleansed["class"] != 7]

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

df_without_7['combined_desc'] = df_without_7['color_desc'] + " " + df_without_7['prod_desc_eng'] + " " + df_without_7['long_desc']
df_without_7['cleaned_combined_desc'] = df_without_7['combined_desc'].apply(clean_stopwords)
df_without_7['cleaned_long_desc'] = df_without_7['long_desc'].apply(clean_stopwords)

# COMMAND ----------

df_without_7.groupby("class").size().plot.bar()

# COMMAND ----------

df_without_7_cleaned = df_without_7[["class","combined_desc"]] #long_desc
df_without_7_cleaned["class"] = df_without_7_cleaned["class"] - 1

# COMMAND ----------

###########Demo##############

# COMMAND ----------

word_counts = df_without_7_cleaned['long_desc'].apply(lambda x: len(x.split()))

# COMMAND ----------


df_without_7_cleaned['long_desc'][3]

# COMMAND ----------

df_without_7_cleaned['long_desc'][7]

# COMMAND ----------

df_without_7_cleaned['long_desc'][13]

# COMMAND ----------

gpt2_input = tokenizer(df_without_7_cleaned['long_desc'].iloc[305], padding="max_length", max_length=100, truncation=True, return_tensors="pt")

# COMMAND ----------

print(gpt2_input['input_ids'])

# COMMAND ----------

print(gpt2_input['input_ids'])

# COMMAND ----------

print(gpt2_input["attention_mask"])

# COMMAND ----------

############Demo end###############

# COMMAND ----------



# COMMAND ----------

np.random.seed(112)
df_train, df_val, df_test = np.split(df_without_7_cleaned.sample(frac=1, random_state=35),[int(0.8*len(df_without_7_cleaned)), int(0.9*len(df_without_7_cleaned))])

print(len(df_train), len(df_val), len(df_test))

# COMMAND ----------

df_train.groupby("class").size().plot.bar()

# COMMAND ----------

from sklearn.utils import resample

df_train_underample0 = df_train.loc[df_train['class']==0]
df_train_underample1 = df_train.loc[df_train['class']==1] # not resmapled
df_train_underample2 = df_train.loc[df_train['class']==2]
df_train_underample3 = df_train.loc[df_train['class']==3]
df_train_underample4 = df_train.loc[df_train['class']==4]
df_train_underample5 = df_train.loc[df_train['class']==5]

for i in range(0, 6): # group 1 is not resampled
    if i != 0 and i != 3:
        exec(f"df_train_underample{i} = resample(df_train_underample{i}, replace=True, n_samples= 130, random_state=123)")
    else:
        exec(f"df_train_underample{i} = resample(df_train_underample{i}, replace=False, n_samples= 130, random_state=123)")

df_train = pd.concat((df_train_underample0,df_train_underample1, df_train_underample2, df_train_underample3, df_train_underample4, df_train_underample5))

# COMMAND ----------

df_train.groupby("class").size().plot.bar()

# COMMAND ----------



# COMMAND ----------

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = df['class'].to_list()
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=75, #128
                                truncation=True,
                                return_tensors="pt") for text in df['combined_desc']] #combined_desc , long_desc
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

# COMMAND ----------

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name, hidden_size = hidden_size)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)

        
    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output

# COMMAND ----------

def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            
            model.zero_grad()

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1)==train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
        with torch.no_grad():
            
            for val_input, val_label in tqdm(val_dataloader):
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                
                output = model(input_id, mask)
                
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1)==val_label).sum().item()
                total_acc_val += acc
                
            print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}")
            
EPOCHS = 20
model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=6, max_seq_len=75, gpt_model_name="gpt2")
LR = 1e-5

# COMMAND ----------

train(model, df_train, df_val, LR, EPOCHS)

# COMMAND ----------

def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

        
    # Tracking variables
    predictions_labels = []
    true_labels = []
    
    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
            
            # add original labels
            true_labels += test_label.cpu().numpy().flatten().tolist()
            # get predicitons to list
            predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return true_labels, predictions_labels
    
true_labels, pred_labels = evaluate(model, df_test)

# COMMAND ----------


# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------

true_labels, pred_labels = evaluate(model, df_val)

# COMMAND ----------

# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------

true_labels, pred_labels = evaluate(model, df_train)

# COMMAND ----------

# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------

