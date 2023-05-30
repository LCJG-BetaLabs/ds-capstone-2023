# Databricks notebook source
# MAGIC %md ### Load data

# COMMAND ----------

#0.1 Data preparation - Get data from source

# Check data from 'pattern_recognition'
import os
import pandas as pd

# LC
container = "data3" 
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df2 = spark.read.format("csv").option("header", "true").load(os.path.join(data_path, "pattern_recognition", "attribute.csv"))

# Convert to pd df, fix "\n" issue in the last column:
df2=df2.toPandas()
df2_piece = df2['atg_code'].iloc[1:]
new_element = pd.Series(['new_value'], index=[0])  # Create a new Series with the new element
df2_piece = pd.concat([ df2_piece,new_element], ignore_index=True)
df2_piece = pd.DataFrame(df2_piece.values, columns=['atg_code'])
result_PR = pd.concat([df2, df2_piece], axis=1, ignore_index=True)
result_PR.columns = list(df2.columns) + ['new_column']
result_PR['img_list_com'] = result_PR['img_list'].str.cat(result_PR['new_column'].astype(str), sep='')
result_PR['img_list_com']=result_PR['img_list_com'].str.replace("\n","")
result_PR=result_PR[result_PR['prod_desc_eng'].notnull()].reset_index(drop=True)
result_PR = result_PR.loc[:,['atg_code', 'prod_desc_eng', 'brand_group_desc', 'brand_desc','atg_bu_desc', 'atg_class_desc','atg_subclass_desc', 'color_desc','compost', 'care', 'size_n_fit', 'long_desc', 'price', 'img_list_com']]
display(result_PR)

# COMMAND ----------

# MAGIC %md ### Add tags

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

train_data.display()

# COMMAND ----------

# Convert predicted label values back to their original string labels
id2label = {i: label for label, i in label2id.items()}
y_pred_str = [id2label[label] for label in y_pred]

# Add predicted labels to test_data dataframe
test_data['predicted_tags'] = y_pred_str

test_data.display()

# COMMAND ----------

# Print classification report
print(classification_report(y_true, y_pred_str))


# COMMAND ----------

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred_str)
print(f'Accuracy: {accuracy:.2f}')

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

#2.0 Transfer learning - preparation

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

# 2.1 Transfer learning - build model with Xception

import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf

# Define the Xception model for feature extraction
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)

# Define the label encoder and scaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Define the list of item IDs and image filenames
item_id = data_tagged['atg_code']  # Replace with your own list of item IDs
img_list = [f"{item}.jpg" for item in item_id]
img_dir = '/databricks/driver'

# Extract features from the images
X = []
y = []
for i, img_file in enumerate(img_list):
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = tf.keras.applications.xception.preprocess_input(np.array([img]))
    vec = base_model.predict(img)
    X.append(vec.ravel())
    item_code = os.path.splitext(img_file)[0]
    label = data_tagged.loc[data_tagged['atg_code'] == item_code, 'tags'].values[0]
    y.append(label)

df=pd.DataFrame({"item id":item_id,"X":X,"Y":y})
# Define the feature and target columns
#feature_cols = ['X']
#target_col = 'Y'


# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Convert the X_train and X_test dataframes to numpy arrays
X_train = np.array([x for x in train_data['X']])
X_test = np.array([x for x in test_data['X']])

# Convert y_train and y_test to numeric labels using LabelEncoder
label_encoder.fit(df["Y"])
y_train = label_encoder.transform(train_data['Y'])
y_test = label_encoder.transform(test_data['Y'])

# Scale the features using StandardScaler
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression classifier on the train set
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Make predictions for each image in the test set
pred_label=[]
for i in range(len(X_test)):
    X_pred = scaler.transform(X_test[i].reshape(1, -1))
    y_pred = clf.predict(X_pred)[0]
    predicted_label = label_encoder.inverse_transform([y_pred])[0]
    pred_label.append(predicted_label)
    actual_label = test_data.iloc[i]['Y']
    item_code = test_data.iloc[i]['item id']
    print(f'{i+1}. Item code: {item_code}; Predicted label: {predicted_label}; Actual label: {actual_label}')
test_data['predicted label']=pred_label
test_data.display()

# COMMAND ----------

# Define a list to store the results
results = []

# Loop over the indices of the test set
for i in range(len(X_test)):
    X_pred = scaler.transform(X_test[i].reshape(1, -1))
    y_pred = clf.predict(X_pred)[0]
    predicted_label_id = y_pred
    predicted_label = label_encoder.inverse_transform([predicted_label_id])[0]
    actual_label = label_encoder.inverse_transform([y_test[i]])[0]
    item_code = data_tagged.iloc[y_test[i]]['atg_code']
    results.append([item_code, actual_label, predicted_label])

# Create a dataframe from the results
results_df = pd.DataFrame(results, columns=['item_code', 'true_label', 'predicted_label'])
pd.set_option('display.max_rows', None)
results_df.display()

# COMMAND ----------

# Observation 1. Dataset being imbalanced lead to poor performance in recognition of particular patterns (e.g. multi_color, stripe, checks). Need to deal with imbalanced data set first

# COMMAND ----------

#0.3 Data-preparation - rebalanced dataset

from imblearn.over_sampling import RandomOverSampler

# Load your data into a pandas DataFrame
df = data_tagged

# Extract the features and target labels
X = df.drop(columns=['tags'])
y = df['tags']

# Print the original class distribution
print('Original class distribution:')
print(y.value_counts())

# Define the random oversampler
oversampler = RandomOverSampler(random_state=42)

# Fit and apply the oversampler to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Print the new class distribution after oversampling
print('New class distribution after oversampling:')
print(pd.Series(y_resampled).value_counts())

# Concatenate the resampled X and y into a new DataFrame
data_tagged_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled)], axis=1)

# COMMAND ----------

#1.1 NLP - Train Bert Model to identify pattern with eng short & long desc, with rebalanced dataset

import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataframe containing the descriptions
# data = pd.read_csv('data.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_tagged_resampled, test_size=0.2, random_state=42)

# Load the pre-trained BERT tokenizer and model
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data_tagged_resampled['tags'].unique()))

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

train_data.display()

# COMMAND ----------

# Convert predicted label values back to their original string labels
id2label = {i: label for label, i in label2id.items()}
y_pred_str = [id2label[label] for label in y_pred]

# COMMAND ----------

# Add predicted labels to test_data dataframe
test_data['predicted_tags'] = y_pred_str

test_data.display()

# COMMAND ----------

# Print classification report
print(classification_report(y_true, y_pred_str))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred_str)
print(f'Accuracy: {accuracy:.2f}')


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
model_path = 'model_resampled.pth'
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

# MAGIC %md ### Download photos

# COMMAND ----------

# 2.2 Transfer learning - build model with Xception

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

# Define the Xception model for feature extraction
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)

# Define the label encoder and scaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Define the list of item IDs and image filenames
item_id = data_tagged['atg_code']  # Replace with your own list of item IDs
img_list = [f"{item}.jpg" for item in item_id]
img_dir = '/databricks/driver'

# Extract features from the images
X = []
y = []
for i, img_file in enumerate(img_list):
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = tf.keras.applications.xception.preprocess_input(np.array([img]))
    vec = base_model.predict(img)
    X.append(vec.ravel())
    item_code = os.path.splitext(img_file)[0]
    label = data_tagged.loc[data_tagged['atg_code'] == item_code, 'tags'].values[0]
    y.append(label)

df=pd.DataFrame({"item id":item_id,"X":X,"Y":y})
# Define the feature and target columns
#feature_cols = ['X']
#target_col = 'Y'

# Extract the features and target labels
X = df.drop(columns=['Y'])
y = df['Y']

# Define the random oversampler
oversampler = RandomOverSampler(random_state=42)

# Fit and apply the oversampler to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Concatenate the resampled X and y into a new DataFrame
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled)], axis=1)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df_resampled, test_size=0.2, random_state=42)

# Convert the X_train and X_test dataframes to numpy arrays
X_train = np.array([x for x in train_data['X']])
X_test = np.array([x for x in test_data['X']])

# Convert y_train and y_test to numeric labels using LabelEncoder
label_encoder.fit(df["Y"])
y_train = label_encoder.transform(train_data['Y'])
y_test = label_encoder.transform(test_data['Y'])

# Scale the features using StandardScaler
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression classifier on the train set
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Make predictions for each image in the test set
pred_label=[]
for i in range(len(X_test)):
    X_pred = scaler.transform(X_test[i].reshape(1, -1))
    y_pred = clf.predict(X_pred)[0]
    predicted_label = label_encoder.inverse_transform([y_pred])[0]
    pred_label.append(predicted_label)
    actual_label = test_data.iloc[i]['Y']
    item_code = test_data.iloc[i]['item id']
    print(f'{i+1}. Item code: {item_code}; Predicted label: {predicted_label}; Actual label: {actual_label}')
test_data['predicted label']=pred_label

# COMMAND ----------

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')

# COMMAND ----------

# Calculate accuracy by label
label_accuracy = {}
for label in label2id.keys():
    label_true = [idx for idx, true_label in enumerate(test_data['Y']) if true_label == label]
    label_pred = [idx for idx, pred_label in enumerate(test_data['predicted label']) if pred_label == label]
    label_intersection = list(set(label_true) & set(label_pred))
    label_accuracy[label] = len(label_intersection) / len(label_true)

# Print accuracy by label
for label, accuracy in label_accuracy.items():
    print(f'Accuracy for label "{label}": {accuracy:.2f}')

# COMMAND ----------

test_data[['item id','Y','predicted label']].display()

# COMMAND ----------

# need to also use the CV model to predict new data

# COMMAND ----------

# observation: after data set is rebalanced, now the accuracy of models improved to a much higher accuracy, but this pose the risk of overfitting, as train test split is performed after resampling. Redo with only training set resampled. 

# COMMAND ----------

# Redo the NLP model

# COMMAND ----------

#0.3 Data-preparation - rebalanced dataset

# COMMAND ----------

# MAGIC %md ### NLP + logistics regression

# COMMAND ----------

#1.3 NLP - Train Bert Model to identify pattern with eng short & long desc, rebalanced train data

import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# Load the dataframe containing the descriptions
# data = pd.read_csv('data.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_tagged, test_size=0.2, random_state=42)

# Load your data into a pandas DataFrame
df = train_data

# Extract the features and target labels
X = df.drop(columns=['tags'])
y = df['tags']

# Print the original class distribution
print('Original class distribution:')
print(y.value_counts())

# Define the random oversampler
oversampler = RandomOverSampler(random_state=42)

# Fit and apply the oversampler to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Print the new class distribution after oversampling
print('New class distribution after oversampling:')
print(pd.Series(y_resampled).value_counts())

# Concatenate the resampled X and y into a new DataFrame
train_data_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled)], axis=1)


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
train_texts = train_data_resampled['concat_desc'].tolist()
train_labels = train_data_resampled['tags'].tolist()

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

# COMMAND ----------

# MAGIC %md ### NLP download excel

# COMMAND ----------

# Add predicted labels to test_data dataframe
test_data['predicted_tags'] = y_pred_str

test_data.display()

# COMMAND ----------

# Print classification report
print(classification_report(y_true, y_pred_str))

# COMMAND ----------

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred_str)
print(f'Accuracy: {accuracy:.2f}')

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
model_path = 'model_resampled.pth'
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

# 2.3 Transfer learning - build model with Xception with rebalanced training set

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

# Define the Xception model for feature extraction
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)

# Define the label encoder and scaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Define the list of item IDs and image filenames
item_id = data_tagged['atg_code']  # Replace with your own list of item IDs
img_list = [f"{item}.jpg" for item in item_id]
img_dir = '/databricks/driver'

# Extract features from the images
X = []
y = []
for i, img_file in enumerate(img_list):
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = tf.keras.applications.xception.preprocess_input(np.array([img]))
    vec = base_model.predict(img)
    X.append(vec.ravel())
    item_code = os.path.splitext(img_file)[0]
    label = data_tagged.loc[data_tagged['atg_code'] == item_code, 'tags'].values[0]
    y.append(label)

df=pd.DataFrame({"item id":item_id,"X":X,"Y":y})
# Define the feature and target columns
#feature_cols = ['X']
#target_col = 'Y'


# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Extract the features and target labels
X = train_data.drop(columns=['Y'])
y = train_data['Y']

# Define the random oversampler
oversampler = RandomOverSampler(random_state=42)

# Fit and apply the oversampler to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Concatenate the resampled X and y into a new DataFrame
train_data_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled)], axis=1)

# Convert X_train and X_test to numpy arrays
X_train = np.array(train_data_resampled['X'].tolist())
X_test = np.array(test_data['X'].tolist())

# Fit the StandardScaler to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the training and test data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the LabelEncoder to the training labels
label_encoder = LabelEncoder()
label_encoder.fit(train_data_resampled['Y'])

# Convert the training and test labels to numeric labels using the fitted LabelEncoder
y_train = label_encoder.transform(train_data_resampled['Y'])
y_test = label_encoder.transform(test_data['Y'])

# Train a logistic regression classifier on the scaled training data
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Evaluate the classifier on the scaled test data
y_pred_enc = clf.predict(X_test_scaled)
pred_label = label_encoder.inverse_transform(y_pred_enc)
accuracy = clf.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy}")

# Make predictions for each image in the test set
for i in range(len(X_test)):
    X_pred = scaler.transform(X_test[i].reshape(1, -1))
    y_pred_enc = clf.predict(X_pred)[0]
    predicted_label = label_encoder.inverse_transform([y_pred_enc])[0]
    actual_label = test_data.iloc[i]['Y']
    item_code = test_data.iloc[i]['item id']
    print(f'{i+1}. Item code: {item_code}; Predicted label: {predicted_label}; Actual label: {actual_label}')
test_data['predicted label']=pred_label

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')

# COMMAND ----------

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')


# COMMAND ----------

# Calculate accuracy by label
label_accuracy = {}
for label in label2id.keys():
    label_true = [idx for idx, true_label in enumerate(test_data['Y']) if true_label == label]
    label_pred = [idx for idx, pred_label in enumerate(test_data['predicted label']) if pred_label == label]
    label_intersection = list(set(label_true) & set(label_pred))
    label_accuracy[label] = len(label_intersection) / len(label_true)

# COMMAND ----------

# Print accuracy by label
for label, accuracy in label_accuracy.items():
    print(f'Accuracy for label "{label}": {accuracy:.2f}')

# COMMAND ----------

# As above is having very low accuracy as observed, find that aspect ratio changed with the code, retry with aspect ratio maintained

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

# Define the Xception model for feature extraction
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)

# Define the label encoder and scaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Define the list of item IDs and image filenames
item_id = data_tagged['atg_code']  # Replace with your own list of item IDs
img_list = [f"{item}.jpg" for item in item_id]
img_dir = '/databricks/driver'

# Extract features from the images
X = []
Y = []
for i, img_file in enumerate(img_list):
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > 1:
        # landscape orientation
        new_width = int(aspect_ratio * 150)
        img = cv2.resize(img, (new_width, 150), interpolation=cv2.INTER_AREA)
        left_padding = max((150 - new_width) // 2, 0)
        right_padding = max(150 - new_width - left_padding, 0)
        img = cv2.copyMakeBorder(img, 0, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # portrait orientation
        new_height = int(150 / aspect_ratio)
        img = cv2.resize(img, (150, new_height), interpolation=cv2.INTER_AREA)
        top_padding = max((150 - new_height) // 2, 0)
        bottom_padding = max(150 - new_height - top_padding, 0)
        img = cv2.copyMakeBorder(img, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    height, width = img.shape[:2]
    if height > width:
        y = (height - width) // 2
        img = img[y:y+width, :]
    else:
        x = (width - height) // 2
        img = img[:, x:x+height]
    img = tf.keras.applications.xception.preprocess_input(np.array([img]))
    vec = base_model.predict(img)
    X.append(vec.ravel())
    item_code = os.path.splitext(img_file)[0]
    label = data_tagged.loc[data_tagged['atg_code'] == item_code, 'tags'].values[0]
    Y.append(label)

df=pd.DataFrame({"item id":item_id,"X":X,"Y":Y})
# Define the feature and target columns
#feature_cols = ['X']
#target_col = 'Y'


# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Extract the features and target labels
X = train_data.drop(columns=['Y'])
y = train_data['Y']

# Define the random oversampler
oversampler = RandomOverSampler(random_state=42)

# Fit and apply the oversampler to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Concatenate the resampled X and y into a new DataFrame
train_data_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled)], axis=1)

# Convert X_train and X_test to numpy arrays
X_train = np.array(train_data_resampled['X'].tolist())
X_test = np.array(test_data['X'].tolist())

# Fit the StandardScaler to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the training and test data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the LabelEncoder to the training labels
label_encoder = LabelEncoder()
label_encoder.fit(train_data_resampled['Y'])

# Convert the training and test labels to numeric labels using the fitted LabelEncoder
y_train = label_encoder.transform(train_data_resampled['Y'])
y_test = label_encoder.transform(test_data['Y'])

# Train a logistic regression classifier on the scaled training data
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Evaluate the classifier on the scaled test data
y_pred_enc = clf.predict(X_test_scaled)
pred_label = label_encoder.inverse_transform(y_pred_enc)
accuracy = clf.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy}")

# Make predictions for each image in the test set
for i in range(len(X_test)):
    X_pred = scaler.transform(X_test[i].reshape(1, -1))
    y_pred_enc = clf.predict(X_pred)[0]
    predicted_label = label_encoder.inverse_transform([y_pred_enc])[0]
    actual_label = test_data.iloc[i]['Y']
    item_code = test_data.iloc[i]['item id']
    print(f'{i+1}. Item code: {item_code}; Predicted label: {predicted_label}; Actual label: {actual_label}')
test_data['predicted label']=pred_label

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')

# COMMAND ----------

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')


# COMMAND ----------

# As performance is disappointing, change to use ResNet50V2 instead and check accuracy

# COMMAND ----------

#3. Transfer learning - CV model with ResNet50V2

import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler

# Define the Xception model for feature extraction
base_model = keras.applications.ResNet50(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)

# Define the label encoder and scaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Define the list of item IDs and image filenames
item_id = data_tagged['atg_code']  # Replace with your own list of item IDs
img_list = [f"{item}.jpg" for item in item_id]
img_dir = '/databricks/driver'

# Extract features from the images
X = []
Y = []
for i, img_file in enumerate(img_list):
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > 1:
        # landscape orientation
        new_width = int(aspect_ratio * 150)
        img = cv2.resize(img, (new_width, 150), interpolation=cv2.INTER_AREA)
        left_padding = max((150 - new_width) // 2, 0)
        right_padding = max(150 - new_width - left_padding, 0)
        img = cv2.copyMakeBorder(img, 0, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # portrait orientation
        new_height = int(150 / aspect_ratio)
        img = cv2.resize(img, (150, new_height), interpolation=cv2.INTER_AREA)
        top_padding = max((150 - new_height) // 2, 0)
        bottom_padding = max(150 - new_height - top_padding, 0)
        img = cv2.copyMakeBorder(img, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    height, width = img.shape[:2]
    if height > width:
        y = (height - width) // 2
        img = img[y:y+width, :]
    else:
        x = (width - height) // 2
        img = img[:, x:x+height]
    img = tf.keras.applications.xception.preprocess_input(np.array([img]))
    vec = base_model.predict(img)
    X.append(vec.ravel())
    item_code = os.path.splitext(img_file)[0]
    label = data_tagged.loc[data_tagged['atg_code'] == item_code, 'tags'].values[0]
    Y.append(label)

df=pd.DataFrame({"item id":item_id,"X":X,"Y":Y})
# Define the feature and target columns
#feature_cols = ['X']
#target_col = 'Y'


# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Extract the features and target labels
X = train_data.drop(columns=['Y'])
y = train_data['Y']

# Define the random oversampler
oversampler = RandomOverSampler(random_state=42)

# Fit and apply the oversampler to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Concatenate the resampled X and y into a new DataFrame
train_data_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled)], axis=1)

# Convert X_train and X_test to numpy arrays
X_train = np.array(train_data_resampled['X'].tolist())
X_test = np.array(test_data['X'].tolist())

# Fit the StandardScaler to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the training and test data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the LabelEncoder to the training labels
label_encoder = LabelEncoder()
label_encoder.fit(train_data_resampled['Y'])

# Convert the training and test labels to numeric labels using the fitted LabelEncoder
y_train = label_encoder.transform(train_data_resampled['Y'])
y_test = label_encoder.transform(test_data['Y'])

# Train a logistic regression classifier on the scaled training data
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Evaluate the classifier on the scaled test data
y_pred_enc = clf.predict(X_test_scaled)
pred_label = label_encoder.inverse_transform(y_pred_enc)
accuracy = clf.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy}")

# Make predictions for each image in the test set
for i in range(len(X_test)):
    X_pred = scaler.transform(X_test[i].reshape(1, -1))
    y_pred_enc = clf.predict(X_pred)[0]
    predicted_label = label_encoder.inverse_transform([y_pred_enc])[0]
    actual_label = test_data.iloc[i]['Y']
    item_code = test_data.iloc[i]['item id']
    print(f'{i+1}. Item code: {item_code}; Predicted label: {predicted_label}; Actual label: {actual_label}')
test_data['predicted label']=pred_label

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')

# COMMAND ----------

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')


# COMMAND ----------

# MAGIC %md ### VGG16 + logistics regression

# COMMAND ----------

#3. Transfer learning - CV model with VGG16

import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler

# Define the Xception model for feature extraction
base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)

# Define the label encoder and scaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Define the list of item IDs and image filenames
item_id = data_tagged['atg_code']  # Replace with your own list of item IDs
img_list = [f"{item}.jpg" for item in item_id]
img_dir = '/databricks/driver'

# Extract features from the images
X = []
Y = []
for i, img_file in enumerate(img_list):
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > 1:
        # landscape orientation
        new_width = int(aspect_ratio * 150)
        img = cv2.resize(img, (new_width, 150), interpolation=cv2.INTER_AREA)
        left_padding = max((150 - new_width) // 2, 0)
        right_padding = max(150 - new_width - left_padding, 0)
        img = cv2.copyMakeBorder(img, 0, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # portrait orientation
        new_height = int(150 / aspect_ratio)
        img = cv2.resize(img, (150, new_height), interpolation=cv2.INTER_AREA)
        top_padding = max((150 - new_height) // 2, 0)
        bottom_padding = max(150 - new_height - top_padding, 0)
        img = cv2.copyMakeBorder(img, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    height, width = img.shape[:2]
    if height > width:
        y = (height - width) // 2
        img = img[y:y+width, :]
    else:
        x = (width - height) // 2
        img = img[:, x:x+height]
    img = tf.keras.applications.xception.preprocess_input(np.array([img]))
    vec = base_model.predict(img)
    X.append(vec.ravel())
    item_code = os.path.splitext(img_file)[0]
    label = data_tagged.loc[data_tagged['atg_code'] == item_code, 'tags'].values[0]
    Y.append(label)

df=pd.DataFrame({"item id":item_id,"X":X,"Y":Y})
# Define the feature and target columns
#feature_cols = ['X']
#target_col = 'Y'


# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Extract the features and target labels
X = train_data.drop(columns=['Y'])
y = train_data['Y']

# Define the random oversampler
oversampler = RandomOverSampler(random_state=42)

# Fit and apply the oversampler to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Concatenate the resampled X and y into a new DataFrame
train_data_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled)], axis=1)

# Convert X_train and X_test to numpy arrays
X_train = np.array(train_data_resampled['X'].tolist())
X_test = np.array(test_data['X'].tolist())

# Fit the StandardScaler to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the training and test data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the LabelEncoder to the training labels
label_encoder = LabelEncoder()
label_encoder.fit(train_data_resampled['Y'])

# Convert the training and test labels to numeric labels using the fitted LabelEncoder
y_train = label_encoder.transform(train_data_resampled['Y'])
y_test = label_encoder.transform(test_data['Y'])

# Train a logistic regression classifier on the scaled training data
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Evaluate the classifier on the scaled test data
y_pred_enc = clf.predict(X_test_scaled)
pred_label = label_encoder.inverse_transform(y_pred_enc)
accuracy = clf.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy}")

# Make predictions for each image in the test set
for i in range(len(X_test)):
    X_pred = scaler.transform(X_test[i].reshape(1, -1))
    y_pred_enc = clf.predict(X_pred)[0]
    predicted_label = label_encoder.inverse_transform([y_pred_enc])[0]
    actual_label = test_data.iloc[i]['Y']
    item_code = test_data.iloc[i]['item id']
    print(f'{i+1}. Item code: {item_code}; Predicted label: {predicted_label}; Actual label: {actual_label}')
test_data['predicted label']=pred_label

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')

# COMMAND ----------

# first time running the code

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')

# COMMAND ----------

# second time running the code

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')

# COMMAND ----------

# third time running the code

from sklearn.metrics import classification_report

# Print classification report
print(classification_report(test_data['Y'],test_data['predicted label']))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['Y'],test_data['predicted label'])
print(f'Accuracy: {accuracy:.2f}')

# COMMAND ----------

#Save model and predict new data

# Save the model to a file
model_path = 'model_resampled.pth'
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

# MAGIC %md ### CV download excel

# COMMAND ----------

# Add predicted labels to test_data dataframe
test_data=test_data[['item id','Y','predicted label']]
test_data.display()

# COMMAND ----------

# MAGIC %md ### CV apply model to new data

# COMMAND ----------

# Define the list of item IDs and image filenames
item_id = data_untagged['atg_code']  # Replace with your own list of item IDs
img_list = [f"{item}.jpg" for item in item_id]
img_dir = '/databricks/driver'

X = []
for i, img_file in enumerate(img_list):
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > 1:
        # landscape orientation
        new_width = int(aspect_ratio * 150)
        img = cv2.resize(img, (new_width, 150), interpolation=cv2.INTER_AREA)
        left_padding = max((150 - new_width) // 2, 0)
        right_padding = max(150 - new_width - left_padding, 0)
        img = cv2.copyMakeBorder(img, 0, 0, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # portrait orientation
        new_height = int(150 / aspect_ratio)
        img = cv2.resize(img, (150, new_height), interpolation=cv2.INTER_AREA)
        top_padding = max((150 - new_height) // 2, 0)
        bottom_padding = max(150 - new_height - top_padding, 0)
        img = cv2.copyMakeBorder(img, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    height, width = img.shape[:2]
    if height > width:
        y = (height - width) // 2
        img = img[y:y+width, :]
    else:
        x = (width - height) // 2
        img = img[:, x:x+height]
    img = tf.keras.applications.xception.preprocess_input(np.array([img]))
    vec = base_model.predict(img)
    X.append(vec.ravel())
    item_code = os.path.splitext(img_file)[0]


df_new_data=pd.DataFrame({"item id":item_id,"X":X})

X_new = np.array(df_new_data['X'].tolist())

# Fit the StandardScaler to the training data
scaler = StandardScaler()
scaler.fit(X_new)

X_new_scaled = scaler.transform(X_new)


y_pred_new = clf.predict(X_new_scaled)
pred_new_label = label_encoder.inverse_transform(y_pred_new)

df_new_data['predicted label']=pred_new_label


df_new_data[["X",'predicted label']].display()



# COMMAND ----------

df_new_data[["item id",'predicted label']].display()

# COMMAND ----------

pwd

# COMMAND ----------

# MAGIC %md ### Loading cfg

# COMMAND ----------

# Define the URL of the file to download
url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'

# Define the local file path to save the downloaded file
local_path = '/databricks/driver/yolov3.cfg'

# Download the file using wget
import os
os.system(f'wget {url} -O {local_path}')

# COMMAND ----------

# MAGIC %md ### Loading weight

# COMMAND ----------

# Define the URL of the file to download
url = 'https://pjreddie.com/media/files/yolov3.weights'

# Define the local file path to save the downloaded file
local_path = '/databricks/driver/yolov3.weights'

# Download the file using wget
import os
os.system(f'wget {url} -O {local_path}')

# COMMAND ----------

# MAGIC %md ### Install open cv

# COMMAND ----------

!pip install opencv-python

# COMMAND ----------

#multi-color model, baseline with a few samples

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Define the YOLO model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define the classifier
clf = LogisticRegression()

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define a function to classify the image as multi-color or not
def classify_image(image):
    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # Get the cropped image of the women top
            cropped_image = get_cropped_image(image, (x, y, w, h))
            # Calculate the color histogram
            hist = calc_color_histogram(cropped_image)
            # Classify the image as multi-color or not
            if hist.std() > threshold:
                return "multi-color"
            else:
                return "not multi-color"
    else:
        return "women top not found"

# Define the input path
#input_path = 'databricks/drivers/'

# Define the dataset
dataset = ['BVR800.jpg', 'BWE389.jpg', 'BVV870.jpg', 'BVO533.jpg', 'BVI489.jpg']

# Define the labels
labels = ['multi-color', 'not multi-color', 'multi-color', 'not multi-color', 'not multi-color']

# Encode the labels as integers
le = LabelEncoder()
le_trans = le.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, le_trans, test_size=0.4, random_state=42)

# Define lists to store the feature vectors and labels
X_train_features = []
y_train_labels = []

# Calculate the feature vector for each image in the training set
for i in range(len(X_train)):
    # Read the image
    image = cv2.imread(X_train[i])
    # Classify the image as multi-color or not
    label = classify_image(image)
    # Add the label to the list of labels
    y_train_labels.append(label)
    # Calculate the color histogram
    hist = calc_color_histogram(image)
    # Add the feature vector to the list of feature vectors
    X_train_features.append(hist)

# Train the logistic regression classifier
clf.fit(X_train_features, y_train)

# Define a function to predict the color of a women top
def predict_color(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Calculate the color histogram
    hist = calc_color_histogram(image)
    # Make a prediction using the logistic regression classifier
    prediction = clf.predict([hist])[0]
    # Decode the label
    label = le.inverse_transform([prediction])[0]
    return label

# Test the classifier on the testing set
y_pred = []
for i in range(len(X_test)):
    # Make a prediction
    prediction = predict_color(X_test[i])
    y_pred.append(prediction)
    # Print the predicted label and the true label
    print("Predicted label:", prediction)
    print("True label:", le.inverse_transform([y_test[i]])[0])

# Print the classification report
print(classification_report(le.inverse_transform(y_test), y_pred))

# COMMAND ----------

#multi-color model, baseline with a few samples (test to edit function)

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Define the YOLO model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define the classifier
clf = LogisticRegression()

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define a function to classify the image as multi-color or not
def classify_image(image):
    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        print("idxs>0")
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # Get the cropped image of the women top
            cropped_image = get_cropped_image(image, (x, y, w, h))
            # Calculate the color histogram
            hist = calc_color_histogram(cropped_image)
            # Classify the image as multi-color or not
            if hist.std() > threshold:
                return "multi-color",idxs
            else:
                return "not multi-color",idxs
    else:
        print("idxs=0")
        return "women top not found",idxs==0

# Define the input path
#input_path = 'databricks/drivers/'

# Define the dataset
dataset = ['BVR800.jpg', 'BWE389.jpg', 'BVV870.jpg', 'BVO533.jpg', 'BVI489.jpg']

# Define the labels
labels = ['multi-color', 'not multi-color', 'multi-color', 'not multi-color', 'not multi-color']

# Encode the labels as integers 
le = LabelEncoder()
le_trans = le.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, le_trans, test_size=0.4, random_state=42)

# Define lists to store the feature vectors and labels
X_train_features = []
y_train_labels = []

# Calculate the feature vector for each image in the training set
for i in range(len(X_train)):
    # Read the image
    image = cv2.imread(X_train[i])
    # Classify the image as multi-color or not
    label,idxs = classify_image(image)
    # Add the label to the list of labels
    y_train_labels.append(label)
    # Calculate the color histogram
    hist = calc_color_histogram(image)
    # Add the feature vector to the list of feature vectors
    X_train_features.append(hist)

# Train the logistic regression classifier
clf.fit(X_train_features, y_train)

# Define a function to predict the color of a women top
def predict_color(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Calculate the color histogram
    hist = calc_color_histogram(image)
    # Make a prediction using the logistic regression classifier
    prediction = clf.predict([hist])[0]
    # Decode the label
    label = le.inverse_transform([prediction])[0]
    return label

# Test the classifier on the testing set
y_pred = []
for i in range(len(X_test)):
    # Make a prediction
    prediction = predict_color(X_test[i])
    y_pred.append(prediction)
    # Print the predicted label and the true label
    print("Predicted label:", prediction)
    print("True label:", le.inverse_transform([y_test[i]])[0])

# Print the classification report
print(classification_report(le.inverse_transform(y_test), y_pred))

# COMMAND ----------

ls

# COMMAND ----------

# MAGIC %md ### Multi-color model (version 1) - logit classifier, incorrect, not use

# COMMAND ----------

# Multi-color model, redo the above to extend to entire dataset

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN

# Define the YOLO model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define the classifier
clf = LogisticRegression()

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define a function to classify the image as multi-color or not
def classify_image(image):
    print('classify_image')
    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id < len(classes) and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        print("idxs>0")
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # Get the cropped image of the women top 
            cropped_image = get_cropped_image(image, (x, y, w, h))
            # Calculate the color histogram
            if len(cropped_image>0):
                hist = calc_color_histogram(cropped_image)
                print("cropped_image")
            else: 
                hist = calc_color_histogram(image)
                print("image")

            # Classify the image as multi-color or not
            if hist.std() > threshold:
                return "multi-color"
            else:
                return "not multi-color"
    else:
    # Calculate the color histogram
        print("idxs=0")
        
        hist = calc_color_histogram(image)
        print("image")
        # Classify the image as multi-color or not
        if hist.std() > threshold:
            return "multi-color"
        else:
            return "not multi-color"

# Define the input path
#input_path = 'databricks/drivers/'

# Define a function to predict the color of a women top
def predict_color(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Calculate the color histogram
    hist = calc_color_histogram(image)
    # Make a prediction using the logistic regression classifier
    prediction = clf.predict([hist])[0]
    # Decode the label
    label = le.inverse_transform([prediction])[0]
    return label


# Define the YOLO model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define the classifier
clf = LogisticRegression()

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define the input path
#input_path = 'databricks/drivers/'

# Define a function to predict the color of a women top
def predict_color(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Calculate the color histogram
    hist = calc_color_histogram(image)
    # Make a prediction using the logistic regression classifier
    prediction = clf.predict([hist])[0]
    # Decode the label
#    label = le.inverse_transform([prediction])[0]
    return prediction

# Define the dataset
dataset=[]
for i in data_tagged['atg_code']:
    a=i+'.jpg'
    dataset.append(a)

# Define the labels
data_tagged['tags_multi'] = np.where(data_tagged['tags'] == 'multi_color', 'multi_color', 'not multi-color')
labels = data_tagged['tags_multi'] 

# Encode the labels as integers
le = LabelEncoder()
le_trans = le.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, le_trans, test_size=0.2, random_state=42)

# Define lists to store the feature vectors and labels
X_train_features = []
y_train_labels = []

# Calculate the feature vector for each image in the training set
for i in range(len(X_train)):
    # Read the image
    image = cv2.imread(X_train[i])
    # Classify the image as multi-color or not
    label = classify_image(image)
    # Add the label to the list of labels
    y_train_labels.append(label)
    # Calculate the color histogram
    hist = calc_color_histogram(image)
    # Add the feature vector to the list of feature vectors
    X_train_features.append(hist)
	 
#adasyn = ADASYN(n_neighbors=3)
#X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_features, y_train_labels)

# Train the logistic regression classifier on the resampled training set
clf.fit(X_train_features, y_train_labels)


# Test the classifier on the testing set
y_pred = []
for i in range(len(X_test)):
    # Make a prediction
    prediction = predict_color(X_test[i])
    y_pred.append(prediction)
    # Print the predicted label and the true label
    print("Predicted label:", prediction)
    print("True label:", le.inverse_transform([y_test[i]])[0])

# Print the classification report
print(classification_report(le.inverse_transform(y_test), y_pred))

# COMMAND ----------

# MAGIC %md ### Multi-color model (version 2) - logit classifier, oversampling

# COMMAND ----------

# Multi-color model, redo the above to extend to entire dataset, over-sampling

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN

# Define the YOLO model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define the classifier
clf = LogisticRegression()

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define a function to classify the image as multi-color or not
def classify_image(image):
    print('classify_image')
    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id < len(classes) and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        print("idxs>0")
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # Get the cropped image of the women top 
            cropped_image = get_cropped_image(image, (x, y, w, h))
            # Calculate the color histogram
            if len(cropped_image > 0):
                hist = calc_color_histogram(cropped_image)
                modified_hist = np.delete(hist, [256, 512, -1])
                print("cropped_image")
                return modified_hist
            else: 
                hist = calc_color_histogram(image)
                modified_hist = np.delete(hist, [256, 512, -1])
                print("image")
                return modified_hist

    else:
    # Calculate the color histogram
        print("idxs=0")        
        hist = calc_color_histogram(image)
        modified_hist = np.delete(hist, [256, 512, -1])
        print("image")
        return modified_hist

# Define the input path
#input_path = 'databricks/drivers/'

# Define a function to predict the color of a women top
def predict_color(hist):

    # Make a prediction using the logistic regression classifier
    prediction = clf.predict([hist])[0]
    # Decode the label
    label = le.inverse_transform([prediction])[0]
    return label


# Define the YOLO model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define the classifier
clf = LogisticRegression()

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define the input path
#input_path = 'databricks/drivers/'


# Define the dataset
dataset=[]
for i in data_tagged['atg_code']:
    a=i+'.jpg'
    dataset.append(a)

# Define the labels
data_tagged['tags_multi'] = np.where(data_tagged['tags'] == 'multi_color', 'multi_color', 'not multi-color')
labels = data_tagged['tags_multi'] 

# Encode the labels as integers
le = LabelEncoder()
le_trans = le.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, le_trans, test_size=0.2, random_state=58)

# Define lists to store the feature vectors and labels
X_train_features = []
y_train_labels = []

# Calculate the feature vector for each image in the training set
for i in range(len(X_train)):
    # Read the image
    image = cv2.imread(X_train[i])
    hist = classify_image(image)
    # Add the label to the list of labels
    #y_train_labels.append(label)
    # Calculate the color histogram
    #hist = calc_color_histogram(image)
    # Add the feature vector to the list of feature vectors
    X_train_features.append(hist)
	 
adasyn = ADASYN(n_neighbors=3)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_features, y_train)

# Train the logistic regression classifier on the resampled training set
clf.fit(X_train_resampled, y_train_resampled)


# Test the classifier on the testing set
y_pred = []
for i in range(len(X_test)):
    # Read the image
    image = cv2.imread(X_test[i])
    hist = classify_image(image)
    # Classify the image as multi-color or not
    pred = predict_color(hist)
    y_pred.append(pred)
    
    # Print the predicted label and the true label
    print("Predicted label:", pred)
    print("True label:", le.inverse_transform([y_test[i]])[0])

# Print the classification report
print(classification_report(le.inverse_transform(y_test), y_pred))

# COMMAND ----------

pd.DataFrame([X_test,le.inverse_transform(y_test), y_pred]).T.display()

# COMMAND ----------

# MAGIC %md ### Multi-color model (version 3) - Logit classifier, undersampling

# COMMAND ----------

# Multi-color model, redo the above to extend to entire dataset, under-sampling

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler

# Define the YOLO model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define the classifier
clf = LogisticRegression()

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define a function to classify the image as multi-color or not
def classify_image(image):
    print('classify_image')
    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id < len(classes) and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        print("idxs>0")
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # Get the cropped image of the women top 
            cropped_image = get_cropped_image(image, (x, y, w, h))
            # Calculate the color histogram
            if len(cropped_image > 0):
                hist = calc_color_histogram(cropped_image)
                modified_hist = np.delete(hist, [256, 512, -1])
                print("cropped_image")
                return modified_hist
            else: 
                hist = calc_color_histogram(image)
                modified_hist = np.delete(hist, [256, 512, -1])
                print("image")
                return modified_hist

    else:
    # Calculate the color histogram
        print("idxs=0")        
        hist = calc_color_histogram(image)
        modified_hist = np.delete(hist, [256, 512, -1])
        print("image")
        return modified_hist


# Define the input path
#input_path = 'databricks/drivers/'

# Define a function to predict the color of a women top
def predict_color(hist):

    # Make a prediction using the logistic regression classifier
    prediction = clf.predict([hist])[0]
    # Decode the label
    label = le.inverse_transform([prediction])[0]
    return label


# Define the YOLO model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define the classifier
clf = LogisticRegression()

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define the input path
#input_path = 'databricks/drivers/'


# Define the dataset
dataset=[]
for i in data_tagged['atg_code']:
    a=i+'.jpg'
    dataset.append(a)

# Define the labels
data_tagged['tags_multi'] = np.where(data_tagged['tags'] == 'multi_color', 'multi_color', 'not multi-color')
labels = data_tagged['tags_multi'] 

# Encode the labels as integers
le = LabelEncoder()
le_trans = le.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, le_trans, test_size=0.2, random_state=42)

# Define lists to store the feature vectors and labels
X_train_features = []
y_train_labels = []

# Calculate the feature vector for each image in the training set
for i in range(len(X_train)):
    # Read the image
    image = cv2.imread(X_train[i])
    hist = classify_image(image)
    # Add the label to the list of labels
    #y_train_labels.append(label)
    # Calculate the color histogram
    #hist = calc_color_histogram(image)
    # Add the feature vector to the list of feature vectors
    X_train_features.append(hist)

rus = RandomUnderSampler(random_state=58)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_features, y_train)


# Train the logistic regression classifier on the resampled training set
clf.fit(X_train_resampled, y_train_resampled)


# Test the classifier on the testing set
y_pred = []
for i in range(len(X_test)):
    # Read the image
    image = cv2.imread(X_test[i])
    hist = classify_image(image)
    # Classify the image as multi-color or not
    pred = predict_color(hist)
    y_pred.append(pred)
    
    # Print the predicted label and the true label
    print("Predicted label:", pred)
    print("True label:", le.inverse_transform([y_test[i]])[0])

# Print the classification report
print(classification_report(le.inverse_transform(y_test), y_pred))

# COMMAND ----------

# MAGIC %md ### Multi-color model (version 4) - CNN

# COMMAND ----------

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from imblearn.over_sampling import RandomOverSampler


# Define the YOLO model
model_yolo = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox): 
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define a function to classify the image as multi-color or not
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

def classify_image(image):
    print('classify_image')
    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model_yolo.setInput(blob)
    output_layers = model_yolo.getUnconnectedOutLayersNames()
    layer_outputs = model_yolo.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id < len(classes) and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # Pad the image to resize it while preserving the aspect ratio
            aspect_ratio = float(w) / h
            target_aspect_ratio = float(TARGET_WIDTH) / TARGET_HEIGHT
            if aspect_ratio > target_aspect_ratio:
                new_w = int(h * target_aspect_ratio)
                x_offset = int((w - new_w) / 2)
                y_offset = 0
            else:
                new_h = int(w / target_aspect_ratio)
                x_offset = 0
                y_offset = int((h - new_h) / 2)
            padded_image = cv2.copyMakeBorder(image[y:y+h, x:x+w], y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Resize the image to the target size
            resized_image = cv2.resize(padded_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    else:
            # Pad the image to resize it while preserving the aspect ratio
            aspect_ratio = float(image.shape[1]) / image.shape[0]
            target_aspect_ratio = float(TARGET_WIDTH) / TARGET_HEIGHT
            if aspect_ratio > target_aspect_ratio:
                new_w = int(image.shape[0] * target_aspect_ratio)
                x_offset = int((image.shape[1] - new_w) / 2)
                y_offset = 0
            else:
                new_h = int(image.shape[1] / target_aspect_ratio)
                x_offset = 0
                y_offset = int((image.shape[0] - new_h) / 2)
            padded_image = cv2.copyMakeBorder(image, y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Resize the image to the target size
            resized_image = cv2.resize(padded_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized_image

# Encode the labels as integers
le = LabelEncoder()
#le_trans = le.fit_transform(labels)


# Define the dataset
dataset=[]
for i in data_tagged['atg_code']:
    a=i+'.jpg'
    dataset.append(a)

# Define the labels
data_tagged['tags_multi'] = np.where(data_tagged['tags'] == 'multi_color', 'multi_color', 'not multi-color')
data_tagged['tags_multi_encode'] = le.fit_transform(data_tagged['tags_multi']) 

# Define the dataset
df = pd.DataFrame({'image_name': dataset, 'tags_multi': data_tagged['tags_multi_encode']})



# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(dataset[:60], le_trans[:60], test_size=0.2, random_state=58)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Extract the features and target labels
X_train = train_data['image_name'].reset_index(drop=True)
y_train = train_data['tags_multi'].reset_index(drop=True)
X_test = test_data['image_name'].reset_index(drop=True)
y_test = test_data['tags_multi'].reset_index(drop=True)
X_train_2d = X_train.values.reshape(-1, 1)
y_train_2d = y_train.values.reshape(-1, 1)



# Define the random oversampler
oversampler = RandomOverSampler(random_state=42)
# Fit and apply the oversampler to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X_train_2d, y_train_2d)


# Define lists to store the feature vectors and labels
X_train_features = []
X_test_features = []

# Calculate the feature vector for each image in the training set
for i in range(len(X_resampled)):
    # Read the image
    image = cv2.imread(X_resampled[i][0])
    processed_image = classify_image(image)
    X_train_features.append(processed_image)

# Calculate the feature vector for each image in the training set
for i in range(len(X_test)):
    # Read the image
    image = cv2.imread(X_test[i])
    processed_image = classify_image(image)
    X_test_features.append(processed_image)

X_train_features = np.array(X_train_features)
X_test_features = np.array(X_test_features)

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_features, y_resampled, epochs=10, batch_size=32, validation_data=(X_test_features, y_test))

# Evaluate the model
y_pred = model.predict(X_test_features)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred_binary)))

# COMMAND ----------

pd.DataFrame((X_test,le.inverse_transform(y_test), le.inverse_transform(y_pred_binary))).T.display()

# COMMAND ----------

print(classification_report(le.inverse_transform(np.array(y_test)), le.inverse_transform(y_pred_binary)))

# COMMAND ----------

# MAGIC %md ### Check model (version 1) - CNN

# COMMAND ----------

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from imblearn.over_sampling import RandomOverSampler


# Define the YOLO model
model_yolo = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# Define a function to get the cropped image of the women top
def get_cropped_image(image, bbox): 
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

# Define a function to calculate the color histogram
def calc_color_histogram(image):
    # Split the image into its color channels
    bgr_planes = cv2.split(image)

    # Calculate the histograms for each color channel
    histograms = []
    for plane in bgr_planes:
        hist = cv2.calcHist([plane], [0], None, [hist_size], hist_range)
        histograms.append(hist)

    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate(histograms).ravel()

    return feature_vector

# Define a function to classify the image as multi-color or not
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

def classify_image(image):
    print('classify_image')
    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model_yolo.setInput(blob)
    output_layers = model_yolo.getUnconnectedOutLayersNames()
    layer_outputs = model_yolo.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id < len(classes) and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # Pad the image to resize it while preserving the aspect ratio
            aspect_ratio = float(w) / h
            target_aspect_ratio = float(TARGET_WIDTH) / TARGET_HEIGHT
            if aspect_ratio > target_aspect_ratio:
                new_w = int(h * target_aspect_ratio)
                x_offset = int((w - new_w) / 2)
                y_offset = 0
            else:
                new_h = int(w / target_aspect_ratio)
                x_offset = 0
                y_offset = int((h - new_h) / 2)
            padded_image = cv2.copyMakeBorder(image[y:y+h, x:x+w], y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Resize the image to the target size
            resized_image = cv2.resize(padded_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    else:
            # Pad the image to resize it while preserving the aspect ratio
            aspect_ratio = float(image.shape[1]) / image.shape[0]
            target_aspect_ratio = float(TARGET_WIDTH) / TARGET_HEIGHT
            if aspect_ratio > target_aspect_ratio:
                new_w = int(image.shape[0] * target_aspect_ratio)
                x_offset = int((image.shape[1] - new_w) / 2)
                y_offset = 0
            else:
                new_h = int(image.shape[1] / target_aspect_ratio)
                x_offset = 0
                y_offset = int((image.shape[0] - new_h) / 2)
            padded_image = cv2.copyMakeBorder(image, y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Resize the image to the target size
            resized_image = cv2.resize(padded_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized_image

# Encode the labels as integers
le = LabelEncoder()
#le_trans = le.fit_transform(labels)


# Define the dataset
dataset=[]
for i in data_tagged['atg_code']:
    a=i+'.jpg'
    dataset.append(a)

# Define the labels
data_tagged['tags_checks'] = np.where(data_tagged['tags'] == 'checks', 'checks', 'not checks')
data_tagged['tags_checks_encode'] = le.fit_transform(data_tagged['tags_checks']) 

# Define the dataset
df = pd.DataFrame({'image_name': dataset, 'tags_checks': data_tagged['tags_checks_encode']})



# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(dataset[:60], le_trans[:60], test_size=0.2, random_state=58)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Extract the features and target labels
X_train = train_data['image_name'].reset_index(drop=True)
y_train = train_data['tags_checks'].reset_index(drop=True)
X_test = test_data['image_name'].reset_index(drop=True)
y_test = test_data['tags_checks'].reset_index(drop=True)
X_train_2d = X_train.values.reshape(-1, 1)
y_train_2d = y_train.values.reshape(-1, 1)



# Define the random oversampler
oversampler = RandomOverSampler(random_state=42)
# Fit and apply the oversampler to the dataset
X_resampled, y_resampled = oversampler.fit_resample(X_train_2d, y_train_2d)


# Define lists to store the feature vectors and labels
X_train_features = []
X_test_features = []

# Calculate the feature vector for each image in the training set
for i in range(len(X_resampled)):
    # Read the image
    image = cv2.imread(X_resampled[i][0])
    processed_image = classify_image(image)
    X_train_features.append(processed_image)

# Calculate the feature vector for each image in the training set
for i in range(len(X_test)):
    # Read the image
    image = cv2.imread(X_test[i])
    processed_image = classify_image(image)
    X_test_features.append(processed_image)

X_train_features = np.array(X_train_features)
X_test_features = np.array(X_test_features)

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_features, y_resampled, epochs=10, batch_size=32, validation_data=(X_test_features, y_test))

# Evaluate the model
y_pred = model.predict(X_test_features)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred_binary)))

# COMMAND ----------

# MAGIC %md ### Multi-color KNN (Version 1)

# COMMAND ----------

# Define the YOLO model
import cv2
model_yolo = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes
classes = ['women top']

# Define the color threshold for multi-color
threshold = 50

# Define the number of bins for the color histogram
hist_size = 256

# Define the range of pixel values for the color histogram
hist_range = (0, 255)

# COMMAND ----------

# Define a function to classify the image as multi-color or not

def crop_image(image):

    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model_yolo.setInput(blob)
    output_layers = model_yolo.getUnconnectedOutLayersNames()
    layer_outputs = model_yolo.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id < len(classes) and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # Pad the image to resize it while preserving the aspect ratio
            x_offset = 0
            y_offset = 0

            cropped_image = cv2.copyMakeBorder(image[y:y+h, x:x+w], y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Resize the image to the target size
            print('cropped_image')

    else:
            cropped_image = image.copy()
            print('original image')

    # Set maximum width for resized image
    max_width = 50

    # Calculate aspect ratio of original image
    if cropped_image is not None and cropped_image.shape:
        aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
        # Calculate new dimensions for resized image
        new_width = min(cropped_image.shape[1], max_width)
        new_height = int(new_width / aspect_ratio)
    else:
        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = min(image.shape[1], max_width)
        new_height = int(new_width / aspect_ratio)

    try:
        # Try resizing the image using the new dimensions
        resized_image = cv2.resize(cropped_image, (new_width, new_height))
    except cv2.error as e:
        # If the resize operation fails, log a warning message and use the original image
        print(f'Warning: {e}. Using original image instead.')
        resized_image = cropped_image

    return resized_image

# COMMAND ----------

# Define a function to classify the image as multi-color or not
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

def crop_image1(image):
    print('crop_image')
    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model_yolo.setInput(blob)
    output_layers = model_yolo.getUnconnectedOutLayersNames()
    layer_outputs = model_yolo.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id < len(classes) and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # Pad the image to resize it while preserving the aspect ratio
            aspect_ratio = float(w) / h
            target_aspect_ratio = float(TARGET_WIDTH) / TARGET_HEIGHT
            if aspect_ratio > target_aspect_ratio:
                new_w = int(h * target_aspect_ratio)
                x_offset = int((w - new_w) / 2)
                y_offset = 0
            else:
                new_h = int(w / target_aspect_ratio)
                x_offset = 0
                y_offset = int((h - new_h) / 2)
            padded_image = cv2.copyMakeBorder(image[y:y+h, x:x+w], y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Resize the image to the target size
            resized_image = cv2.resize(padded_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    else:
            # Pad the image to resize it while preserving the aspect ratio
            aspect_ratio = float(image.shape[1]) / image.shape[0]
            target_aspect_ratio = float(TARGET_WIDTH) / TARGET_HEIGHT
            if aspect_ratio > target_aspect_ratio:
                new_w = int(image.shape[0] * target_aspect_ratio)
                x_offset = int((image.shape[1] - new_w) / 2)
                y_offset = 0
            else:
                new_h = int(image.shape[1] / target_aspect_ratio)
                x_offset = 0
                y_offset = int((image.shape[0] - new_h) / 2)
            padded_image = cv2.copyMakeBorder(image, y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Resize the image to the target size
            resized_image = cv2.resize(padded_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized_image

# COMMAND ----------

# Define a function to classify the image as multi-color or not
#TARGET_WIDTH = 224
#TARGET_HEIGHT = 224

def crop_image0(image):
    print('crop_image')
    # Get the bounding box coordinates of the women top using YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model_yolo.setInput(blob)
    output_layers = model_yolo.getUnconnectedOutLayersNames()
    layer_outputs = model_yolo.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id < len(classes) and classes[class_id] == 'women top':
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cropped_image = image[y:y+h, x:x+w]

    else:
            # Pad the image to resize it while preserving the aspect ratio
            cropped_image=image
    return cropped_image

# COMMAND ----------

def predict_multicolor0(img, min_clusters=2, min_mass=0.05, min_percent=50):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create masks for skin color and white background
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    mask = cv2.bitwise_or(skin_mask, white_mask)
    
    # Apply mask to image to remove skin color and white background
    img_masked = cv2.bitwise_and(img, img, mask=~mask)
    
    # Flatten image
    img_flat = img_masked.reshape((-1, 3))

    # Cast img_flat to np.float32
    img_flat = img_flat.astype(np.float32)

    # Find optimal K using elbow method
    wcss = []
    max_K = 10
    for k in range(1, max_K+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(img_flat)
        wcss.append(kmeans.inertia_)
    diff = np.diff(wcss)
    diff_r = diff[1:] / diff[:-1]
    k_opt = np.argmax(diff_r) + 2

    # K-means clustering with optimal K
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 5
    ret, labels, centers = cv2.kmeans(img_flat, k_opt, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels to original image shape
    labels = labels.reshape(img_masked.shape[:2])

    # Calculate mass for each cluster
    unique, counts = np.unique(labels, return_counts=True)
    mass = dict(zip(unique, counts))

    # Remove background cluster and any cluster with center close to white
    for i in range(k_opt):
        if np.allclose(centers[i], [255, 255, 255], atol=10):
            mass.pop(i, None)
    mass.pop(-1, None)

    # Calculate mass threshold
    mass_threshold = min_mass * img.size / k_opt

    # Count number of clusters with mass above threshold
    num_large_clusters = sum(m > mass_threshold for m in mass.values())

    # Predict if multi-color
    num_clusters = len(mass)
    if num_clusters >= min_clusters and num_large_clusters >= k_opt * min_percent / 100:
        results = 'multi_color'
    else:
        results = 'not multi_color'
    return results,k_opt,num_large_clusters
    #return results

# COMMAND ----------

!pip install webcolors
import webcolors

def predict_multicolor2(img, min_mass=0.0, min_percent=80, max_percent_diff=0.305):
    k = 5
    
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create masks for skin color and white background
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    mask = cv2.bitwise_or(skin_mask, white_mask)
    
    # Apply mask to image to remove skin color and white background
    img_masked = cv2.bitwise_and(img, img, mask=~mask)
    
    # Flatten image
    img_flat = img_masked.reshape((-1, 3))

    # Cast img_flat to np.float32
    img_flat = img_flat.astype(np.float32)

    # K-means clustering with k=6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, labels, centers = cv2.kmeans(img_flat, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels to original image shape
    labels = labels.reshape(img_masked.shape[:2])

    # Calculate mass for each cluster
    unique, counts = np.unique(labels, return_counts=True)
    mass = dict(zip(unique, counts))


    # Remove background cluster and any cluster with center close to white
    for i in range(k):
        if np.allclose(centers[i], [255, 255, 255], atol=10):
            mass.pop(i, None)

    # Sort mass in descending order
    sorted_mass = sorted(mass.values(), reverse=True)

    # Create a list of colors represented by the clusters with the highest masses
    colors = []
    for i in range(k):
        if i in mass:
            rgb = tuple(centers[i].astype(int))
            try:
                color_name = webcolors.rgb_to_name(rgb)
                colors.append(color_name)
            except ValueError:
                pass

    # Sort colors by mass in descending order
    sorted_colors = sorted(colors, key=lambda x: colors.count(x), reverse=True)


    # Calculate sum of mass for the 3 largest clusters
    sum_mass = sum(sorted_mass[:3])

    # Calculate percentage of pixels captured by the 3 largest clusters
    percent_mass = sum_mass / img_masked.size * 100

    # Check if there are at least 3 non-zero clusters
    num_clusters = len(sorted_mass)
    if num_clusters >= 3:
        # Check if the 3 largest clusters capture at least 80% of pixels
        if percent_mass <= min_percent:
            # Check if the 3 largest clusters are of comparable size
 #           largest_mass = sorted_mass[:2]
            percent_diff = abs(sorted_mass[0] - sorted_mass[1]) / sorted_mass[1]
            if percent_diff <= max_percent_diff:
                results = 'multi_color'
                # Return results, number of clusters, and sum of mass for the 3 largest clusters
#                return results, k, sum_mass

    # If none of the conditions are met, classify as not multi-color
    results = 'not multi_color'

    # Return results, number of clusters, and sum of mass for the 3 smallest clusters
    return results, k, img_masked.size, percent_mass,percent_diff,sorted_mass,sorted_colors,sorted_mass[0] ,sorted_mass[1]

# COMMAND ----------

def predict_multicolor1(img, min_mass=0.0, min_percent=80, max_percent_diff=0.305):
    k = 6
    
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create masks for skin color and white background
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    mask = cv2.bitwise_or(skin_mask, white_mask)
    
    # Apply mask to image to remove skin color and white background
    img_masked = cv2.bitwise_and(img, img, mask=~mask)
    
    # Flatten image
    img_flat = img_masked.reshape((-1, 3))

    # Cast img_flat to np.float32
    img_flat = img_flat.astype(np.float32)

    # K-means clustering with k=6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, labels, centers = cv2.kmeans(img_flat, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels to original image shape
    labels = labels.reshape(img_masked.shape[:2])

    # Calculate mass for each cluster
    unique, counts = np.unique(labels, return_counts=True)
    mass = dict(zip(unique, counts))

    # Remove background cluster and any cluster with center close to white
    for i in range(k):
        if np.allclose(centers[i], [255, 255, 255], atol=10):
            mass.pop(i, None)

    # Sort mass in descending order
    sorted_mass = sorted(mass.values(), reverse=True)

    # Calculate sum of mass for the 3 largest clusters
    sum_mass = sum(sorted_mass[:3])

    # Calculate percentage of pixels captured by the 3 largest clusters
    percent_mass = sum_mass / img_masked.size * 100

    # Check if there are at least 3 non-zero clusters
    num_clusters = len(sorted_mass)
    if num_clusters >= 3:
        # Check if the 3 largest clusters capture at least 80% of pixels
        if percent_mass <= min_percent:
            # Check if the 3 largest clusters are of comparable size
 #           largest_mass = sorted_mass[:2]
            percent_diff = abs(sorted_mass[0] - sorted_mass[1]) / sorted_mass[1]
            if percent_diff <= max_percent_diff:
                results = 'multi_color'
                # Return results, number of clusters, and sum of mass for the 3 largest clusters
#                return results, k, sum_mass

    # If none of the conditions are met, classify as not multi-color
    results = 'not multi_color'

    # Return results, number of clusters, and sum of mass for the 3 smallest clusters
    return results, k, img_masked.size, percent_mass,percent_diff,sorted_mass,sorted_mass[0] ,sorted_mass[1]

# COMMAND ----------

def predict_multicolor2(img, min_mass=0.0, min_percent=80, max_percent_diff=0.305):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create masks for skin color and white background
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    mask = cv2.bitwise_or(skin_mask, white_mask)
    
    # Apply mask to image to remove skin color and white background
    img_masked = cv2.bitwise_and(img, img, mask=~mask)
    
    # Flatten image
    img_flat = img_masked.reshape((-1, 3))

    # Cast img_flat to np.float32
    img_flat = img_flat.astype(np.float32)

    # Elbow method to find optimal value of k
    wcss = []
    for i in range(1, 7):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=5, random_state=0)
        kmeans.fit(img_flat)
        wcss.append(kmeans.inertia_)

    # Choose k where the decrease in WCSS starts to level off
    k = 3  # default value
    for i in range(1, len(wcss)):
        if (wcss[i] - wcss[i-1]) < 0.05 * wcss[0]:
            k = i
            break
    
    # K-means clustering with learned k
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=5, random_state=0)
    kmeans.fit(img_flat)

    # Reshape labels to original image shape
    labels = kmeans.labels_.reshape(img_masked.shape[:2])

    # Calculate mass for each cluster
    unique, counts = np.unique(labels, return_counts=True)
    mass = dict(zip(unique, counts))

    # Remove background cluster and any cluster with center close to white
    centers = kmeans.cluster_centers_
    for i in range(k):
        if np.allclose(centers[i], [255, 255, 255], atol=10):
            mass.pop(i, None)

    # Sort mass in descending order
    sorted_mass = sorted(mass.values(), reverse=True)
    if len(sorted_mass)==1:
        sorted_mass[1]==0

    # Calculate sum of mass for the 3 largest clusters
    sum_mass = sum(sorted_mass[:3])

    # Calculate percentage of pixels captured by the 3 largest clusters
    percent_mass = sum_mass / img_masked.size * 100

    # Check if there are at least 3 non-zero clusters
    num_clusters = len(sorted_mass)
    percent_diff = abs(sorted_mass[0] - sorted_mass[1]) / sorted_mass[1]
    if num_clusters >= 3:
        # Check if the 3 largest clusters capture at least 80% of pixels
        if percent_mass <= min_percent:
            # Check if the 3 largest clusters are of comparable size           
            if percent_diff <= max_percent_diff:
                results = 'multi_color'
                # Return results, number of clusters, and sum of mass for the 3 largest clusters
 #               return results, k, sum_mass

    # If none of the conditions are met, classify as not multi-color
    results = 'not multi_color'

    # Return results, number of clusters, and sum of mass for the 3 smallest clusters
    return results, k, img_masked.size, percent_mass, percent_diff, sorted_mass[0],sorted_mass[1]

# COMMAND ----------

def predict_multicolor_version1(img, min_clusters=5, min_per_mass=30, min_percent_diff=3):
    #Convert image to HSV color space
    try: 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        #Create masks for skin color and white background
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        mask = cv2.bitwise_or(skin_mask, white_mask)
        #Apply mask to image to remove skin color and white background
        img_masked = cv2.bitwise_and(img, img, mask=~mask)
    except: 
        img_masked = img

    
    # Flatten image
    img_flat = img_masked.reshape((-1, 3))

    # Cast img_flat to np.float32
    img_flat = img_flat.astype(np.float32)

    # Find optimal K using elbow method
    wcss = []
    max_K = 8
    for k in range(1, max_K+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=5, random_state=0)
        kmeans.fit(img_flat)
        wcss.append(kmeans.inertia_)
    diff = np.diff(wcss)
    diff_r = diff[1:] / diff[:-1]
    k_opt = np.argmax(diff_r) + 2

    # K-means clustering with optimal K
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 5
    ret, labels, centers = cv2.kmeans(img_flat, k_opt, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels to original image shape
    labels = labels.reshape(img_masked.shape[:2])

    # Calculate mass for each cluster
    unique, counts = np.unique(labels, return_counts=True)
    mass = dict(zip(unique, counts))

    # Remove background cluster and any cluster with center close to white
    for i in range(k_opt):
        if np.allclose(centers[i], [255, 255, 255], atol=10):
            mass.pop(i, None)
    mass.pop(-1, None)

    # Sort mass in descending order
    sorted_mass = sorted(mass.values(), reverse=True)

    # Calculate sum of mass for the 3 largest clusters
    sum_mass = sum(sorted_mass[:3])

    # Calculate percentage of pixels captured by the 3 largest clusters
    percent_mass = sum_mass / img_masked.size * 100

    # Calculate percentage difference
    percent_diff = abs(sorted_mass[0]-sorted_mass[1]) / sorted_mass[1]

    # Calculate mass threshold
 #   mass_threshold = min_mass * img.size / k_opt

    # Count number of clusters with mass above threshold
 #   num_large_clusters = sum(m > mass_threshold for m in mass.values())

    # Predict if multi-color
    num_clusters = len(mass)
    score=0
    if num_clusters <= min_clusters:
        score=score+1
    if percent_mass <=min_per_mass:
        score=score+1
    if percent_diff<=min_percent_diff:
        score=score+1
    if score >=2: 
        results = 'multi_color'
    else:
        results = 'not multi_color'
    return results, k_opt, img_masked.size, percent_mass,percent_diff,sorted_mass,sorted_mass[0] ,sorted_mass[1],score

#return results, k_opt, img_masked.size, percent_mass,percent_diff,sorted_mass,sorted_mass[0] ,sorted_mass[1]


# COMMAND ----------

def predict_multicolor_version2(img, min_clusters=3, min_per_mass=30, min_percent_diff=3):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create masks for skin color and white background
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    mask = cv2.bitwise_or(skin_mask, white_mask)
    
    # Apply mask to image to remove skin color and white background
    img_masked = cv2.bitwise_and(img, img, mask=~mask)
    
    # Flatten image
    img_flat = img_masked.reshape((-1, 3))

    # Cast img_flat to np.float32
    img_flat = img_flat.astype(np.float32)

    # Find optimal K using elbow method
    wcss = []
    max_K = 6
    for k in range(1, max_K+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=5, random_state=0)
        kmeans.fit(img_flat)
        wcss.append(kmeans.inertia_)
    diff = np.diff(wcss)
    diff_r = diff[1:] / diff[:-1]
    k_opt = np.argmax(diff_r) + 2

    # K-means clustering with optimal K
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 5
    ret, labels, centers = cv2.kmeans(img_flat, k_opt, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels to original image shape
    labels = labels.reshape(img_masked.shape[:2])

    # Calculate mass for each cluster
    unique, counts = np.unique(labels, return_counts=True)
    mass = dict(zip(unique, counts))

    # Remove background cluster and any cluster with center close to white
    for i in range(k_opt):
        if np.allclose(centers[i], [255, 255, 255], atol=10):
            mass.pop(i, None)
    mass.pop(-1, None)

    # Sort mass in descending order
    sorted_mass = sorted(mass.values(), reverse=True)

    # Calculate sum of mass for the 3 largest clusters
    sum_mass = sum(sorted_mass[:3])

    # Calculate percentage of pixels captured by the 3 largest clusters
    percent_mass = sum_mass / img_masked.size * 100

    # Calculate percentage difference
    percent_diff = abs(sorted_mass[0]-sorted_mass[1]) / sorted_mass[1]

    # Calculate mass threshold
 #   mass_threshold = min_mass * img.size / k_opt

    # Count number of clusters with mass above threshold
 #   num_large_clusters = sum(m > mass_threshold for m in mass.values())

    # Predict if multi-color
    num_clusters = len(mass)
    score=0
    if num_clusters <= min_clusters:
        score=score+1
    if percent_mass <=min_per_mass:
        score=score+1
    if percent_diff<=min_percent_diff:
        score=score+1
    if score >=2: 
        results = 'multi_color'
    else:
        results = 'not multi_color'
    return results, k_opt, img_masked.size, percent_mass,percent_diff,sorted_mass,sorted_mass[0] ,sorted_mass[1],score

#return results, k_opt, img_masked.size, percent_mass,percent_diff,sorted_mass,sorted_mass[0] ,sorted_mass[1]


# COMMAND ----------

#predict_multicolor_version1
import cv2
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from imblearn.over_sampling import RandomOverSampler
from sklearn.cluster import KMeans

# Define skin color and white background ranges in HSV color space
skin_lower = np.array([0, 10, 60], dtype=np.uint8)
skin_upper = np.array([20, 255, 255], dtype=np.uint8)
white_lower = np.array([0, 0, 180], dtype=np.uint8)
white_upper = np.array([255, 20, 255], dtype=np.uint8)

# Define a function to classify the image as multi-color or not
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

# Define the dataset
dataset=[]
for i in data_tagged['atg_code']:
    a=i+'.jpg'
    dataset.append(a)

# Encode the labels as integers
le = LabelEncoder()

# Define the labels
data_tagged['tags_multi_color'] = np.where(data_tagged['tags'] == 'multi_color', 'multi_color', 'not multi_color')
#data_tagged['tags_multi_color_encode'] = le.fit_transform(data_tagged['tags_multi_color']) 

# Define the dataset
df = pd.DataFrame({'image_name': dataset, 'tags_multi_color': data_tagged['tags_multi_color']})
#df=df.iloc[0:30,:]
#df = df[df['tags_multi_color']=='multi_color']

pred_label=[]
no_cluster=[]
masked_image_size=[]
per_mass=[]
percent_diff=[]
smass=[]
M0=[]
M1=[]
scores=[]
# Calculate the feature vector for each image in the training set
for z in df['image_name']:
    # Read the image
    image = cv2.imread(z)
    processed_image = crop_image(image)
    result, no_clusters, mask_img_sizes, percent_of_mass, perc_diff, smas, m0, m1,score = predict_multicolor_version1(processed_image)

#    result=predict_multicolor(processed_image)
    pred_label.append(result)
    no_cluster.append(no_clusters)
    masked_image_size.append(mask_img_sizes)
    per_mass.append(percent_of_mass)
    percent_diff.append(perc_diff)
    smass.append(smas)
 #   scol.append(col)
    M0.append(m0)
    M1.append(m1)
    scores.append(score)
print(classification_report(df['tags_multi_color'], pred_label))

# COMMAND ----------

#predict_multicolor_version1
df1=pd.DataFrame({"image_name":df['image_name'],"Actual":df['tags_multi_color'], "Pred":pred_label,"no_cluster":no_cluster,"Masked Image Size":masked_image_size,"per mass":per_mass,"per diff":percent_diff,"Sorted Mass":smass,"M0":M0,"M1":M1,"score":scores})
df1 = df1.reset_index()
df1.display()

# COMMAND ----------

#predict_multicolor_version2

import cv2
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from imblearn.over_sampling import RandomOverSampler
from sklearn.cluster import KMeans

# Define skin color and white background ranges in HSV color space
skin_lower = np.array([0, 10, 60], dtype=np.uint8)
skin_upper = np.array([20, 255, 255], dtype=np.uint8)
white_lower = np.array([0, 0, 180], dtype=np.uint8)
white_upper = np.array([255, 20, 255], dtype=np.uint8)

# Define a function to classify the image as multi-color or not
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

# Define the dataset
dataset=[]
for i in data_tagged['atg_code']:
    a=i+'.jpg'
    dataset.append(a)

# Encode the labels as integers
le = LabelEncoder()

# Define the labels
data_tagged['tags_multi_color'] = np.where(data_tagged['tags'] == 'multi_color', 'multi_color', 'not multi_color')
#data_tagged['tags_multi_color_encode'] = le.fit_transform(data_tagged['tags_multi_color']) 

# Define the dataset
df = pd.DataFrame({'image_name': dataset, 'tags_multi_color': data_tagged['tags_multi_color']})
df=df.iloc[:20,:]
#df = df[df['tags_multi_color']=='multi_color']

pred_label=[]
no_cluster=[]
masked_image_size=[]
per_mass=[]
percent_diff=[]
smass=[]
scol=[]
M0=[]
M1=[]
scores=[]
# Calculate the feature vector for each image in the training set
for z in df['image_name']:
    # Read the image
    image = cv2.imread(z)
    processed_image = crop_image(image)
    result, no_clusters, mask_img_sizes, percent_of_mass, perc_diff, smas, m0, m1,score = predict_multicolor_version2(processed_image)

#    result=predict_multicolor(processed_image)
    pred_label.append(result)
    no_cluster.append(no_clusters)
    masked_image_size.append(mask_img_sizes)
    per_mass.append(percent_of_mass)
    percent_diff.append(perc_diff)
    smass.append(smas)
 #   scol.append(col)
    M0.append(m0)
    M1.append(m1)
    scores.append(score)
print(classification_report(df['tags_multi_color'], pred_label))

# COMMAND ----------

#predict_multicolor_version2
df1=pd.DataFrame({"image_name":df['image_name'],"Actual":df['tags_multi_color'], "Pred":pred_label,"no_cluster":no_cluster,"Masked Image Size":masked_image_size,"per mass":per_mass,"per diff":percent_diff,"Sorted Mass":smass,"M0":M0,"M1":M1,"Scores":scores})
df1 = df1.reset_index()
df1.display()