# Databricks notebook source
import os
import pandas as pd
import csv

# LC
container = "data3" 
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.format("csv").option("header", "true").load(os.path.join(data_path, "pattern_recognition", "attribute.csv"))


# Convert to pd df, fix "\n" issue in the last column:
df=df.toPandas()
df_piece = df['atg_code'].iloc[1:]
new_element = pd.Series(['new_value'], index=[0])  # Create a new Series with the new element
df_piece = pd.concat([ df_piece,new_element], ignore_index=True)
df_piece = pd.DataFrame(df_piece.values, columns=['atg_code'])
result_PR = pd.concat([df, df_piece], axis=1, ignore_index=True)
result_PR.columns = list(df.columns) + ['new_column']
result_PR['img_list_com'] = result_PR['img_list'].str.cat(result_PR['new_column'].astype(str), sep='')
result_PR['img_list_com']=result_PR['img_list_com'].str.replace("\n","")
result_PR=result_PR[result_PR['prod_desc_eng'].notnull()].reset_index(drop=True)
result_PR = result_PR.loc[:,['atg_code', 'prod_desc_eng', 'brand_group_desc', 'brand_desc','atg_bu_desc', 'atg_class_desc','atg_subclass_desc', 'color_desc','compost', 'care', 'size_n_fit', 'long_desc', 'price', 'img_list_com']]
display(result_PR)

# COMMAND ----------

#0.2 Data preparation - Attach tag to data, create extra column with 'product desc+long desc), separate data to data_tagged and data_untagged

tags = ['plain', 'graphic_print', 'multi_color', 'word_print', 'graphic_print', 'graphic_print', 'graphic_print', 'word_print', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'word_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'graphic_print', 'graphic_print', 'untagged', 'plain', 'plain', 'graphic_print', 'word_print', 'plain', 'graphic_print', 'word_print', 'plain', 'plain', 'plain', 'graphic_print', 'plain', 'word_print', 'plain', 'stripe', 'stripe', 'graphic_print', 'graphic_print', 'untagged', 'stripe', 'graphic_print', 'untagged', 'untagged', 'word_print', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'word_print', 'plain', 'graphic_print', 'word_print', 'untagged', 'plain', 'plain', 'plain', 'word_print', 'plain', 'plain', 'word_print', 'graphic_print', 'graphic_print', 'graphic_print', 'word_print', 'multi_color', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'stripe', 'untagged', 'stripe', 'plain', 'plain', 'multi_color', 'untagged', 'graphic_print', 'plain', 'plain', 'graphic_print', 'stripe', 'plain', 'multi_color', 'checks', 'plain', 'graphic_print', 'plain', 'plain', 'graphic_print', 'untagged', 'untagged', 'untagged', 'plain', 'untagged', 'untagged', 'untagged', 'untagged', 'stripe', 'stripe', 'untagged', 'untagged', 'graphic_print', 'untagged', 'graphic_print', 'word_print', 'graphic_print', 'plain', 'stripe', 'word_print', 'plain', 'plain', 'multi_color', 'plain', 'plain', 'graphic_print', 'graphic_print', 'plain', 'plain', 'plain', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'stripe', 'plain', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'word_print', 'graphic_print', 'word_print', 'untagged', 'graphic_print', 'untagged', 'plain', 'graphic_print', 'plain', 'plain', 'plain', 'checks', 'graphic_print', 'word_print', 'stripe', 'untagged', 'word_print', 'untagged', 'untagged', 'untagged', 'graphic_print', 'untagged', 'word_print', 'untagged', 'untagged', 'plain', 'plain', 'word_print', 'plain', 'untagged', 'plain', 'plain', 'word_print', 'plain', 'graphic_print', 'untagged', 'graphic_print', 'stripe', 'graphic_print', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'word_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'plain', 'stripe', 'untagged', 'plain', 'stripe', 'untagged', 'graphic_print', 'plain', 'plain', 'plain', 'checks', 'untagged', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'multi_color', 'untagged', 'stripe', 'untagged', 'plain', 'untagged', 'word_print', 'graphic_print', 'plain', 'plain', 'plain', 'word_print', 'checks', 'plain', 'plain', 'plain', 'plain', 'graphic_print', 'graphic_print', 'plain', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'word_print', 'graphic_print', 'graphic_print', 'untagged', 'word_print', 'word_print', 'graphic_print', 'untagged', 'plain', 'plain', 'stripe', 'stripe', 'plain', 'stripe', 'plain', 'word_print', 'plain', 'plain', 'plain', 'plain', 'plain', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'untagged', 'graphic_print', 'word_print', 'untagged', 'stripe', 'untagged', 'plain', 'plain', 'graphic_print', 'graphic_print', 'plain', 'graphic_print', 'plain', 'stripe', 'graphic_print', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'untagged', 'graphic_print', 'plain', 'graphic_print', 'untagged', 'untagged', 'untagged', 'graphic_print', 'graphic_print', 'graphic_print', 'word_print', 'checks', 'word_print', 'plain', 'stripe', 'plain', 'plain', 'word_print', 'multi_color', 'untagged', 'plain', 'graphic_print', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'multi_color', 'untagged', 'untagged', 'untagged', 'untagged', 'graphic_print', 'untagged', 'plain', 'plain', 'plain', 'plain', 'plain', 'untagged', 'multi_color', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'multi_color', 'multi_color', 'graphic_print', 'stripe', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'word_print', 'untagged', 'graphic_print', 'untagged', 'plain', 'plain', 'plain', 'plain', 'plain', 'plain', 'graphic_print', 'plain', 'plain', 'stripe', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'word_print', 'plain', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'plain', 'graphic_print', 'untagged', 'plain', 'untagged', 'multi_color', 'plain', 'plain', 'stripe', 'multi_color', 'graphic_print', 'graphic_print', 'graphic_print', 'plain', 'word_print', 'plain', 'plain', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'word_print', 'untagged', 'graphic_print', 'graphic_print', 'plain', 'untagged', 'graphic_print', 'plain', 'plain', 'multi_color', 'word_print', 'plain', 'graphic_print', 'plain', 'graphic_print', 'plain', 'plain', 'plain', 'stripe', 'untagged', 'multi_color', 'untagged', 'untagged', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'graphic_print', 'word_print', 'untagged', 'graphic_print', 'multi_color', 'untagged', 'untagged', 'untagged', 'multi_color', 'plain', 'untagged', 'checks', 'stripe', 'plain', 'plain', 'multi_color', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'stripe', 'untagged', 'plain', 'plain', 'word_print', 'plain', 'graphic_print', 'word_print', 'plain', 'stripe', 'untagged', 'untagged', 'graphic_print', 'stripe', 'graphic_print', 'stripe', 'graphic_print', 'untagged', 'graphic_print', 'stripe', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'word_print', 'untagged', 'untagged', 'untagged', 'checks', 'multi_color', 'plain', 'untagged', 'untagged', 'graphic_print', 'plain', 'plain', 'multi_color', 'plain', 'plain', 'plain', 'plain', 'plain', 'stripe', 'plain', 'untagged', 'graphic_print', 'untagged', 'word_print', 'untagged', 'word_print', 'word_print', 'untagged', 'graphic_print', 'untagged', 'multi_color', 'graphic_print', 'untagged', 'word_print', 'untagged', 'word_print', 'word_print', 'graphic_print', 'plain', 'graphic_print', 'untagged', 'plain', 'plain', 'plain', 'plain', 'graphic_print', 'graphic_print', 'stripe', 'plain', 'stripe', 'word_print', 'untagged', 'untagged', 'untagged', 'untagged', 'word_print', 'word_print', 'untagged', 'plain', 'checks', 'word_print', 'graphic_print', 'plain', 'plain', 'untagged', 'untagged', 'graphic_print', 'stripe', 'untagged', 'untagged', 'untagged', 'word_print', 'stripe', 'graphic_print', 'graphic_print', 'untagged', 'graphic_print', 'graphic_print', 'stripe', 'word_print', 'multi_color', 'untagged', 'plain', 'plain', 'plain', 'plain', 'multi_color', 'plain', 'checks', 'graphic_print', 'plain', 'plain', 'word_print', 'graphic_print', 'graphic_print', 'graphic_print', 'stripe', 'stripe', 'untagged', 'untagged', 'graphic_print', 'untagged', 'graphic_print', 'graphic_print', 'untagged', 'stripe', 'stripe', 'plain', 'untagged', 'word_print', 'plain', 'graphic_print', 'word_print', 'plain', 'graphic_print', 'untagged', 'plain', 'graphic_print', 'plain', 'stripe', 'stripe', 'stripe', 'untagged', 'graphic_print', 'multi_color', 'word_print', 'graphic_print', 'word_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'stripe', 'multi_color', 'stripe', 'graphic_print', 'multi_color', 'untagged', 'untagged', 'stripe', 'plain', 'untagged', 'untagged', 'plain', 'plain', 'plain', 'plain', 'word_print', 'word_print', 'graphic_print', 'stripe', 'graphic_print', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged', 'stripe', 'plain', 'stripe', 'plain', 'word_print', 'plain', 'word_print', 'graphic_print', 'plain', 'plain', 'word_print', 'plain', 'graphic_print', 'word_print', 'graphic_print', 'plain', 'graphic_print', 'plain', 'graphic_print', 'untagged', 'untagged', 'untagged', 'graphic_print', 'checks', 'untagged', 'untagged', 'untagged', 'checks', 'untagged', 'untagged', 'untagged', 'untagged', 'graphic_print', 'word_print', 'plain', 'graphic_print', 'checks', 'graphic_print', 'untagged', 'untagged', 'checks', 'untagged', 'untagged', 'untagged', 'plain', 'plain', 'plain', 'plain', 'graphic_print', 'plain', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'untagged', 'stripe', 'checks', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'graphic_print', 'multi_color', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'untagged', 'multi_color', 'untagged', 'plain', 'plain', 'plain', 'plain', 'plain', 'multi_color', 'plain', 'word_print', 'word_print', 'plain', 'stripe', 'untagged', 'graphic_print', 'untagged', 'untagged', 'graphic_print', 'plain', 'untagged', 'untagged', 'untagged', 'word_print', 'graphic_print', 'graphic_print', 'stripe', 'graphic_print', 'word_print', 'plain', 'plain', 'plain', 'stripe', 'untagged', 'untagged', 'untagged', 'graphic_print', 'word_print', 'untagged', 'untagged', 'untagged', 'untagged', 'plain', 'multi_color', 'graphic_print', 'plain', 'untagged', 'stripe', 'plain', 'plain', 'plain', 'graphic_print', 'graphic_print', 'graphic_print', 'plain', 'plain', 'word_print', 'untagged', 'plain', 'checks', 'graphic_print', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'untagged', 'stripe', 'multi_color', 'graphic_print', 'plain', 'plain', 'plain', 'graphic_print', 'plain', 'word_print', 'word_print', 'plain', 'untagged', 'graphic_print', 'plain', 'multi_color', 'graphic_print', 'graphic_print', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'word_print', 'untagged', 'untagged', 'untagged', 'untagged', 'untagged','stripe', 'untagged', 'plain', 'untagged', 'word_print', 'untagged', 'plain', 'graphic_print', 'graphic_print', 'plain', 'word_print', 'plain', 'plain', 'untagged', 'stripe', 'stripe', 'untagged', 'graphic_print', 'graphic_print', 'untagged', 'graphic_print', 'untagged', 'graphic_print', 'untagged', 'graphic_print', 'graphic_print', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'word_print', 'untagged', 'untagged', 'untagged', 'untagged', 'word_print', 'multi_color', 'plain', 'plain', 'plain', 'graphic_print', 'word_print', 'plain', 'multi_color', 'graphic_print', 'word_print', 'word_print', 'word_print', 'stripe', 'graphic_print', 'plain', 'plain', 'graphic_print', 'word_print', 'untagged', 'untagged', 'graphic_print', 'graphic_print', 'untagged', 'untagged', 'graphic_print', 'untagged', 'untagged', 'untagged', 'graphic_print', 'multi_color', 'untagged', 'untagged', 'stripe', 'untagged', 'graphic_print', 'graphic_print', 'plain', 'plain', 'multi_color', 'stripe', 'plain', 'untagged', 'plain', 'plain', 'graphic_print']
data = result_PR
data['concat_desc'] =data['prod_desc_eng']  +" " + data['long_desc']
data['tags']=tags
data_tagged=data[data['tags']!='untagged']
data_untagged=data[data['tags']=='untagged']
print(data_tagged.shape)
print(data_untagged.shape)




# COMMAND ----------

#1.0 NLP - Train Bert Model to identify pattern with eng short & long desc

import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataframe containing the descriptions
# data = pd.read_csv('data.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_tagged, test_size=0.2, random_state=42)

# Load the pre-trained BERT tokenizer and model
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data_tagged['tags'].unique()))

# Define the training parameters
batch_size = 32
num_epochs = 5
learning_rate = 2e-5

# Define the device (GPU or CPU) to use for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the optimizer and loss function
optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Define the data loader for the training set
train_texts = train_data['concat_desc'].tolist()
train_labels = train_data['tags'].tolist()

# Convert string labels to integer labels
label2id = {label: i for i, label in enumerate(data_tagged['tags'].unique())}
train_labels = [label2id[label] for label in train_labels]

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the model on the training set
model.to(device)
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1} loss: {loss.item()}')

# Evaluate the model on the testing set
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for i, row in test_data.iterrows():
        text = row['concat_desc']
        label = row['tags']
        encoding = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = np.argmax(logits.detach().cpu().numpy())

        y_true.append(label)
        y_pred.append(predicted_label)

# COMMAND ----------

print(train_data.shape)
print(test_data.shape)

# COMMAND ----------

# Convert predicted label values back to their original string labels
id2label = {i: label for label, i in label2id.items()}
y_pred_str = [id2label[label] for label in y_pred]

# Add predicted labels to test_data dataframe
test_data['predicted_tags'] = y_pred_str

test_data.display()

# COMMAND ----------

print(classification_report(y_true, y_pred_str))


# COMMAND ----------

# Calculate accuracy by label
label_accuracy = {}
for label in label2id.keys():
    label_true = [idx for idx, true_label in enumerate(y_true) if true_label == label]
    label_pred = [idx for idx, pred_label in enumerate(y_pred_str) if pred_label == label]
    label_intersection = list(set(label_true) & set(label_pred))
    label_accuracy[label] = len(label_intersection) / len(label_true)

# Print accuracy by label
for label, accuracy in label_accuracy.items():
    print(f'Accuracy for label "{label}": {accuracy:.2f}')

# COMMAND ----------

#Save model and predict new data

# Save the model to a file
model_path = 'model.pth'
torch.save(model.state_dict(), model_path)

# Load the saved model
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data_tagged['tags'].unique()))
model.load_state_dict(torch.load(model_path))
model.eval()

# Tokenize the new data and use the saved model to predict labels
new_texts = data_untagged['concat_desc'].tolist()
new_encodings = tokenizer(new_texts, truncation=True, padding=True, return_tensors='pt')
with torch.no_grad():
    inputs = {'input_ids': new_encodings['input_ids'].to(device),
              'attention_mask': new_encodings['attention_mask'].to(device)}
    outputs = model(**inputs)
    preds = outputs.logits.argmax(dim=1).cpu().numpy()
    new_pred_str = [id2label[idx] for idx in preds]

# Add the predicted labels to the DataFrame
data_untagged['tags'] = new_pred_str

# Print the DataFrame with predicted labels
data_untagged.display()

# COMMAND ----------

!pip install opencv-python
import os
import requests
from PIL import Image
from io import BytesIO
import pandas as pd

def download_image_not_exists(item_id):
    file_name = item_id+'.jpg'
    if not os.path.exists(file_name):
        url = 'https://media.lanecrawford.com/' + item_id[0] + '/' + item_id[1] + '/' + item_id[2] + '/' + item_id + '_in_xl.jpg'
        response = requests.get(url)
        if response.status_code==404:
            return None
        img = Image.open(BytesIO(response.content))
        img.save(file_name)
    return file_name

item_id=data['atg_code']
for i in item_id:
    download_image_not_exists(i)

# COMMAND ----------

import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

display(data)



# COMMAND ----------

