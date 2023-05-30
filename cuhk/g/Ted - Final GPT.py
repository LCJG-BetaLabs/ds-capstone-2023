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

#df1 = pd.read_csv("/dbfs/FileStore/shared_uploads/capstone2023_cuhk_team_g@ijam.onmicrosoft.com/new_tag2a.csv")

# COMMAND ----------

new_key_dict = df1.set_index('atg_code')['tag'].to_dict()

class_mapping = {
    'graphic_print' : 1,
    'multi_color' : 2,
    'word_print' : 3,
    'plain' : 4,
    'checks': 5,
    'stripe': 6,
    'untagged': 7,
    'additional checks': 8,
    'additional diamonds': 9
}

# COMMAND ----------

test_df = pd.read_csv('/dbfs/final_test_data3.csv')
train_df = pd.read_csv('/dbfs/final_train_data3.csv')
val_df = pd.read_csv('/dbfs/final_val_data3.csv')

# COMMAND ----------

#test_df["class"] = test_df["atg_code"].replace(new_key_dict)
#test_df["class"] = test_df["class"].replace(class_mapping)
#test_df["class"] = test_df["class"]-1

# COMMAND ----------

#train_df["class"] = train_df["atg_code"].replace(new_key_dict)
#train_df["class"] = train_df["class"].replace(class_mapping)
#train_df["class"] = train_df["class"]-1

# COMMAND ----------

#val_df["class"] = val_df["atg_code"].replace(new_key_dict)
#val_df["class"] = val_df["class"].replace(class_mapping)
#val_df["class"] = val_df["class"]-1

# COMMAND ----------

test_df = pd.read_csv('/dbfs/df_test_method3.csv')
train_df = pd.read_csv('/dbfs/df_train_method3.csv')
val_df = pd.read_csv('/dbfs/df_val_method3.csv')

train_df["class"] = train_df["class"] - 1
val_df["class"] = val_df["class"] - 1
test_df["class"] = test_df["class"] - 1

# COMMAND ----------



# COMMAND ----------

train_df.head()

# COMMAND ----------

############Ted part#######################

# COMMAND ----------

#gpt_word_embedding_df = pd.read_csv('/dbfs/final_gpt_ebd_df_100_04271345.csv')

# COMMAND ----------

gpt_word_embedding_df_care = pd.read_csv('/dbfs/final_gpt_ebd_df_care_43.csv')
gpt_word_embedding_df_color = pd.read_csv('/dbfs/final_gpt_ebd_df_color_desc_4.csv',)
gpt_word_embedding_df_prod_desc_eng = pd.read_csv('/dbfs/final_gpt_ebd_df_prod_desc_eng_43.csv')
gpt_word_embedding_df_long_desc = pd.read_csv('/dbfs/final_gpt_ebd_df_long_desc_100.csv')

# COMMAND ----------

gpt_word_embedding_df_care.head()

# COMMAND ----------

gpt_word_embedding_df_color.head()

# COMMAND ----------

gpt_word_embedding_df_prod_desc_eng.head()

# COMMAND ----------

gpt_word_embedding_df = pd.merge(gpt_word_embedding_df_care,gpt_word_embedding_df_color,left_on = ['0','1'], right_on= ['0','1'])

# COMMAND ----------

gpt_word_embedding_df = pd.merge(gpt_word_embedding_df,gpt_word_embedding_df_prod_desc_eng,left_on = ['0','1'], right_on= ['0','1'])

# COMMAND ----------

gpt_word_embedding_df = pd.merge(gpt_word_embedding_df,gpt_word_embedding_df_long_desc,left_on = ['0','1'], right_on= ['0','1'])

# COMMAND ----------

gpt_word_embedding_df

# COMMAND ----------

gpt_train_emd_df = pd.merge(train_df[["atg_code"]],gpt_word_embedding_df,left_on="atg_code", right_on="0")
gpt_val_emd_df = pd.merge(val_df[["atg_code"]],gpt_word_embedding_df,left_on="atg_code", right_on="0")
gpt_test_emd_df = pd.merge(test_df[["atg_code"]],gpt_word_embedding_df,left_on="atg_code", right_on="0")

# COMMAND ----------

len(gpt_train_emd_df),len(gpt_val_emd_df),len(gpt_test_emd_df)

# COMMAND ----------

gpt_X_train = np.array(gpt_train_emd_df.drop(['atg_code', '0', '1'], axis=1))
gpt_X_val = np.array(gpt_val_emd_df.drop(['atg_code', '0', '1'], axis=1))
gpt_X_test = np.array(gpt_test_emd_df.drop(['atg_code', '0', '1'], axis=1))

# COMMAND ----------

gpt_y_train = np.array(train_df['class'].tolist())
gpt_y_train_transformed = K.utils.to_categorical(gpt_y_train)

# COMMAND ----------

gpt_y_val = np.array(val_df['class'].tolist())
gpt_y_val_transformed = K.utils.to_categorical(gpt_y_val)

# COMMAND ----------

gpt_y_test = np.array(test_df['class'].tolist())
gpt_y_test_transformed = K.utils.to_categorical(gpt_y_test)

# COMMAND ----------

from sklearn.metrics import accuracy_score
from keras.layers import Flatten
EPOCH_SIZE = 50

FILEPATH = "/dbfs/text_model_vted"

# Early stopping  
K.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCH_SIZE/10),
check_point = K.callbacks.ModelCheckpoint(filepath=FILEPATH, monitor="val_loss", mode="min", save_best_only=True,)

# Define neural network architecture
input_shape = (gpt_X_train.shape[1],)
num_classes = gpt_y_train_transformed.shape[1]
model = K.models.Sequential()
model.add(K.layers.Dense(1024, input_shape = input_shape, activation="relu"))
#model.add(K.layers.Dropout(0.2))

#model.add(K.layers.BatchNormalization())
#model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(gpt_X_train, gpt_y_train_transformed, batch_size=8, epochs=EPOCH_SIZE, verbose=1, validation_data=(gpt_X_val, gpt_y_val_transformed),callbacks=[check_point])

model.summary()
#model.save(FILEPATH)

# COMMAND ----------

true_labels

# COMMAND ----------

test_predicted_prob = model.predict(gpt_X_test)
predicted_labels = np.argmax(test_predicted_prob, axis=1)
true_labels = np.argmax(gpt_y_test_transformed, axis = 1)
print(sum(predicted_labels == true_labels) / len(true_labels))


# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------

##########Gen Untagged#########

# COMMAND ----------

gpt_untagged_emd_df = pd.merge(df_cleansed[df_cleansed["class"] == 7][["atg_code"]],gpt_word_embedding_df,left_on="atg_code", right_on="0")

# COMMAND ----------

gpt_untagged_emd_df

# COMMAND ----------

#gpt_X_7 = np.array(df7.drop(['Unnamed: 0', '0', '1'], axis=1))
gpt_X_7 = np.array(gpt_untagged_emd_df.drop(['atg_code', '0', '1'], axis=1))

# COMMAND ----------

untag_predicted_prob = model.predict(gpt_X_7)

# COMMAND ----------

predicted_labels = np.argmax(untag_predicted_prob, axis=1)

# COMMAND ----------

gpt_untagged_emd_df["predicted_lables"] = predicted_labels

# COMMAND ----------

#df7["predicted_lables"][(np.max(untag_predicted_prob, axis=1) < 0.8)] = 10

# COMMAND ----------

untagged_prob_df = pd.DataFrame(untag_predicted_prob, columns= ["Class1","Class2","Class3","Class4","Class5","Class6"])

# COMMAND ----------

output = pd.concat([gpt_untagged_emd_df[["atg_code","predicted_lables"]],untagged_prob_df],axis = 1)

# COMMAND ----------

output[["atg_code","predicted_lables","Class1","Class2","Class3","Class4","Class5","Class6"]].to_csv("/dbfs/untagged_gpt_202304271605.csv", index= False)

# COMMAND ----------

gpt_untagged_emd_df

# COMMAND ----------

pd.read_csv('/dbfs/untagged_gpt_202304271605.csv')

# COMMAND ----------

#################################################Test output###############################################

# COMMAND ----------

gpt_test_emd_df_for_output = gpt_test_emd_df.copy()
untag_predicted_prob = model.predict(gpt_X_test)
predicted_labels = np.argmax(untag_predicted_prob, axis=1)
gpt_test_emd_df_for_output["predicted_lables"] = predicted_labels
untagged_prob_df = pd.DataFrame(untag_predicted_prob, columns= ["Class1","Class2","Class3","Class4","Class5","Class6"])
output = pd.concat([gpt_test_emd_df_for_output[["atg_code","predicted_lables"]],untagged_prob_df],axis = 1)

# COMMAND ----------

output.to_csv("/dbfs/test_result_gpt_202304271605.csv", index= False)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

output

# COMMAND ----------

gpt_test_emd_df_for_output

# COMMAND ----------

