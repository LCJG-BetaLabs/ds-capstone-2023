# Databricks notebook source
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)


import os
df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df_cleansed["class"] = df_cleansed["class"]-1
df_cleansed = df_cleansed.set_index("atg_code")

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

#Keith
Restnet_test_df = pd.read_csv('/dbfs/image_prob_test_rest50.csv')
Restnet_test_df = Restnet_test_df.drop(['Unnamed: 0'],axis = 1)
Restnet_test_df = Restnet_test_df.set_index("atg_code")
Restnet_test_df_without_pred = Restnet_test_df.drop(['pred'],axis = 1)


Restnet_untagged_df = pd.read_csv('/dbfs/image_prob_untagged_rest50.csv')
Restnet_untagged_df = Restnet_untagged_df.drop(['Unnamed: 0'],axis = 1)
Restnet_untagged_df = Restnet_untagged_df.set_index("atg_code")
Restnet_untagged_df_without_pred = Restnet_untagged_df.drop(['pred'],axis = 1)

#Ted

Gpt_test_df = pd.read_csv("/dbfs/test_result_gpt_202304271605.csv")
Gpt_test_df = Gpt_test_df.set_index("atg_code")
Gpt_test_df_without_pred = Gpt_test_df.drop(['predicted_lables'],axis = 1)

Gpt_untagged_df = pd.read_csv('/dbfs/untagged_gpt_202304271605.csv')
Gpt_untagged_df = Gpt_untagged_df.set_index("atg_code")
Gpt_untagged_df_without_pred = Gpt_untagged_df.drop(['predicted_lables'],axis = 1)

#Na 姐

TFIDF_test_df = pd.read_csv('/dbfs/tfidf_test_prob_data.csv')
TFIDF_test_df = TFIDF_test_df.set_index("atg_code")
TFIDF_test_df_without_pred = TFIDF_test_df.drop(['pred'],axis = 1)

TFIDF_untagged_df = pd.read_csv('/dbfs/tfidf_untagged_prob_data.csv')
TFIDF_untagged_df = TFIDF_untagged_df.set_index("atg_code")
TFIDF_untagged_df_without_pred = TFIDF_untagged_df.drop(['pred'],axis = 1)

# COMMAND ----------

Restnet_untagged_df[Restnet_untagged_df["pred"] == 4]

# COMMAND ----------

TFIDF_untagged_df_without_pred.head()

# COMMAND ----------



# COMMAND ----------

Restnet_weight = 0.5
GPT_weight = 0.25
TFIDF_weight = 0.25
Thre = 0.5

# COMMAND ----------

#Restnet_test_df_without_pred_adj = Restnet_test_df_without_pred.copy()
#Restnet_test_df_without_pred_adj["Class1"] = Restnet_test_df_without_pred_adj["Class1"] * Restnet_weight
#Restnet_test_df_without_pred_adj["Class2"] = Restnet_test_df_without_pred_adj["Class2"] * 0
#Restnet_test_df_without_pred_adj["Class3"] = Restnet_test_df_without_pred_adj["Class3"] * 0
#Restnet_test_df_without_pred_adj["Class4"] = Restnet_test_df_without_pred_adj["Class4"] * Restnet_weight
#Restnet_test_df_without_pred_adj["Class5"] = Restnet_test_df_without_pred_adj["Class5"] * Restnet_weight
#Restnet_test_df_without_pred_adj["Class6"] = Restnet_test_df_without_pred_adj["Class6"] * Restnet_weight


# COMMAND ----------

Gpt_test_df_without_pred_adj = Gpt_test_df_without_pred.copy()
Gpt_test_df_without_pred_adj["Class1"] = Gpt_test_df_without_pred_adj["Class1"] * GPT_weight
Gpt_test_df_without_pred_adj["Class2"] = Gpt_test_df_without_pred_adj["Class2"] * 0
Gpt_test_df_without_pred_adj["Class3"] = Gpt_test_df_without_pred_adj["Class3"] * 0
Gpt_test_df_without_pred_adj["Class4"] = Gpt_test_df_without_pred_adj["Class4"] * GPT_weight
Gpt_test_df_without_pred_adj["Class5"] = Gpt_test_df_without_pred_adj["Class5"] * GPT_weight
Gpt_test_df_without_pred_adj["Class6"] = Gpt_test_df_without_pred_adj["Class6"] * GPT_weight


# COMMAND ----------

TFIDF_test_df_without_pred_adj = TFIDF_test_df_without_pred.copy()
TFIDF_test_df_without_pred_adj["Class1"] = TFIDF_test_df_without_pred_adj["Class1"] * TFIDF_weight
TFIDF_test_df_without_pred_adj["Class2"] = TFIDF_test_df_without_pred_adj["Class2"] * 0
TFIDF_test_df_without_pred_adj["Class3"] = TFIDF_test_df_without_pred_adj["Class3"] * 0
TFIDF_test_df_without_pred_adj["Class4"] = TFIDF_test_df_without_pred_adj["Class4"] * TFIDF_weight
TFIDF_test_df_without_pred_adj["Class5"] = TFIDF_test_df_without_pred_adj["Class5"] * TFIDF_weight
TFIDF_test_df_without_pred_adj["Class6"] = TFIDF_test_df_without_pred_adj["Class6"] * TFIDF_weight

# COMMAND ----------

Final_probability_df = ((Restnet_test_df_without_pred*Restnet_weight).add(Gpt_test_df_without_pred*GPT_weight, fill_value=0)).add(TFIDF_test_df_without_pred*TFIDF_weight, fill_value=0)

#Final_probability_df = ((Restnet_test_df_without_pred_adj).add(Gpt_test_df_without_pred_adj, fill_value=0)).add(TFIDF_test_df_without_pred_adj, fill_value=0)

# COMMAND ----------

predicted_labels = np.argmax(np.array(Final_probability_df), axis=1)
check_thre = (np.max(np.array(Final_probability_df), axis=1) < Thre)

# COMMAND ----------

Final_probability_df["pred"] = predicted_labels
Final_probability_df["pred"][check_thre] = None

# COMMAND ----------

final_output = pd.merge(df_cleansed[["class"]],Final_probability_df[["pred"]],left_index= True, right_index= True)

# COMMAND ----------

final_output = final_output.dropna()

# COMMAND ----------

predicted_labels = np.array(final_output['pred'])
true_labels = np.array(final_output["class"])
print(sum(predicted_labels == true_labels) / len(true_labels))


# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------



# COMMAND ----------

Restnet_weight = 0.5
GPT_weight = 0.25
TFIDF_weight = 0.25
Thre = 0.5

# COMMAND ----------

Final_probability_df_untagged = ((Restnet_untagged_df_without_pred*Restnet_weight).add(Gpt_untagged_df_without_pred*GPT_weight, fill_value=0)).add(TFIDF_untagged_df_without_pred*TFIDF_weight, fill_value=0)

#Final_probability_df_untagged = ((Restnet_untagged_df_without_pred_adj).add(Gpt_untagged_df_without_pred_adj, fill_value=0)).add(TFIDF_untagged_df_without_pred_adj, fill_value=0)

# COMMAND ----------

predicted_labels = np.argmax(np.array(Final_probability_df_untagged), axis=1)
check_thre = (np.max(np.array(Final_probability_df_untagged), axis=1) < Thre)

# COMMAND ----------

Final_probability_df_untagged["pred"] = predicted_labels
Final_probability_df_untagged["pred"][check_thre] = 99

# COMMAND ----------

final_output = pd.merge(df_cleansed[["class"]],Final_probability_df_untagged[["pred"]],left_index= True, right_index= True)

# COMMAND ----------

len(final_output)

# COMMAND ----------

final_output.to_csv("/dbfs/final_combine_output_untagged.csv")

# COMMAND ----------

final_output[final_output[#]]

# COMMAND ----------

Final_probability_df_untagged

# COMMAND ----------

Restnet_untagged_df

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df_cleansed["class"] = df_cleansed["class"]-1

final_output = pd.merge(df_cleansed[['atg_code',"class"]],Restnet_test_df_output[['atg_code','pred']],left_on= 'atg_code',right_on= 'atg_code')
predicted_labels = np.array(final_output['pred'])
true_labels = np.array(final_output["class"])
print(sum(predicted_labels == true_labels) / len(true_labels))


# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------



# COMMAND ----------

np.array(Restnet_test_df.drop(['atg_code',"pred"],axis = 1))

# COMMAND ----------

df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df_cleansed["class"] = df_cleansed["class"]-1

Restnet_test_df = pd.read_csv('/dbfs/image_prob_test_v3.csv')
Restnet_test_df = Restnet_test_df.drop(['Unnamed: 0'],axis = 1)
Restnet_test_df_temp = Restnet_test_df.copy()

predicted_labels = np.argmax(np.array(Restnet_test_df.drop(['atg_code',"pred"],axis = 1)), axis=1)
Restnet_test_df_temp["pred"] = predicted_labels

final_output = pd.merge(df_cleansed[['atg_code',"class"]],Restnet_test_df_temp[['atg_code','pred']],left_on= 'atg_code',right_on= 'atg_code')

true_labels = np.array(final_output["class"])
predicted_labels = np.array(final_output["pred"])
print(sum(predicted_labels == true_labels) / len(true_labels))


# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------

df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df_cleansed["class"] = df_cleansed["class"]-1

Gpt_test_df = pd.read_csv("/dbfs/test_result_gpt_202304271605.csv")

final_output = pd.merge(df_cleansed[['atg_code',"class"]],Gpt_test_df[['atg_code','predicted_lables']],left_on= 'atg_code',right_on= 'atg_code')

predicted_labels = np.array(final_output["predicted_lables"])
true_labels = np.array(final_output["class"])
print(sum(predicted_labels == true_labels) / len(true_labels))


# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------

df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df_cleansed["class"] = df_cleansed["class"]-1

TFIDF_test_df = pd.read_csv('/dbfs/tfidf_test_prob_data.csv')

final_output = pd.merge(df_cleansed[['atg_code',"class"]],TFIDF_test_df[['atg_code','pred']],left_on= 'atg_code',right_on= 'atg_code')

predicted_labels = np.array(final_output["pred"])
true_labels = np.array(final_output["class"])
print(sum(predicted_labels == true_labels) / len(true_labels))


# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------

predicted_labels

# COMMAND ----------

df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df_cleansed["class"] = df_cleansed["class"]-1

Gpt_test_df = pd.read_csv("/dbfs/test_result_gpt_202304271605.csv")
Gpt_test_df_temp = pd.read_csv("/dbfs/test_result_gpt_202304271605.csv")


predicted_labels = np.argmax(np.array(Gpt_test_df.drop(['atg_code',"predicted_lables"],axis = 1)), axis=1)
Gpt_test_df_temp["pred"] = predicted_labels

final_output = pd.merge(df_cleansed[['atg_code',"class"]],Gpt_test_df_temp[['atg_code','pred']],left_on= 'atg_code',right_on= 'atg_code')

true_labels = np.array(final_output["class"])

print(sum(predicted_labels == true_labels) / len(true_labels))


# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------

np.argmax(np.array(Gpt_test_df_temp.drop(['atg_code',"predicted_lables"],axis = 1)), axis=1)

# COMMAND ----------

Gpt_test_df_temp

# COMMAND ----------

final_output

# COMMAND ----------

