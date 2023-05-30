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

test_df = pd.read_csv('/dbfs/final_test_data.csv')
train_df = pd.read_csv('/dbfs/final_train_data.csv')
val_df = pd.read_csv('/dbfs/final_val_data.csv')

# COMMAND ----------

test_df = pd.read_csv('/dbfs/df_test_method3.csv')
train_df = pd.read_csv('/dbfs/df_train_method3.csv')
val_df = pd.read_csv('/dbfs/df_val_method3.csv')

# COMMAND ----------

train_df["class"] = train_df["class"] - 1
val_df["class"] = val_df["class"] - 1
test_df["class"] = test_df["class"] - 1

# COMMAND ----------

############Ted part#######################

# COMMAND ----------

gpt_word_embedding_df = pd.read_csv('/dbfs/final_gpt_ebd_df_125_04271345.csv')

# COMMAND ----------

gpt_train_emd_df = pd.merge(train_df[["atg_code"]],gpt_word_embedding_df,left_on="atg_code", right_on="0")
gpt_val_emd_df = pd.merge(val_df[["atg_code"]],gpt_word_embedding_df,left_on="atg_code", right_on="0")
gpt_test_emd_df = pd.merge(test_df[["atg_code"]],gpt_word_embedding_df,left_on="atg_code", right_on="0")

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score


for k in range(6,30,2):
    print(k)
    kmeans = KMeans(n_clusters=k)
    X = gpt_train_emd_df.drop(["atg_code","0","1"],axis = 1)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    score = silhouette_score(X, labels)

    # Print the R-squared statistic
    print("silhouette_score:", score)

# COMMAND ----------

k = 12

print(k)
kmeans = KMeans(n_clusters=k)
X = gpt_train_emd_df.drop(["atg_code","0","1"],axis = 1)
kmeans.fit(X)

# COMMAND ----------

labels = kmeans.labels_

# COMMAND ----------

gpt_train_emd_df["cluster"] = labels

# COMMAND ----------

for i in range(0,(k)):
    plt.figure()  # create a new figure for each chart
    gpt_train_emd_df[gpt_train_emd_df["cluster"] == i].groupby("1").size().plot.bar()
    plt.title(f"Cluster {i}")  # add a title to the chart

# COMMAND ----------



# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

knn = KNeighborsClassifier(n_neighbors=20)

# COMMAND ----------

knn.fit(gpt_train_emd_df.drop(["atg_code","0","1"],axis = 1), gpt_train_emd_df["1"])

# COMMAND ----------

y_pred = knn.predict(gpt_val_emd_df.drop(["atg_code","0","1"],axis = 1))

# COMMAND ----------

true_labels = gpt_val_emd_df["1"]
pred_labels = y_pred

print(sum(pred_labels == true_labels) / len(true_labels))

# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df_embs_original = pd.read_csv(os.path.join("file:/dbfs", "image_embedding_data_new.csv"))

# COMMAND ----------

df_embs_original.drop(["Y","atg_code"])

# COMMAND ----------

df_embs_original

# COMMAND ----------

rest_train_emd_df = pd.merge(train_df[["atg_code"]],df_embs_original,left_on="atg_code", right_on="atg_code")
rest_val_emd_df = pd.merge(val_df[["atg_code"]],df_embs_original,left_on="atg_code", right_on="atg_code")
rest_test_emd_df = pd.merge(test_df[["atg_code"]],df_embs_original,left_on="atg_code", right_on="atg_code")

# COMMAND ----------

df_embs_original = df_embs_original[df_embs_original["Y"] != 7]
df_embs_original = df_embs_original[df_embs_original["Y"] != 8]
df_embs_original = df_embs_original[df_embs_original["Y"] != 9]

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

k = 6

print(k)
kmeans = KMeans(n_clusters=k)
X = df_embs_original.drop(["Y","atg_code"],axis = 1)
kmeans.fit(X)

# COMMAND ----------



# COMMAND ----------

labels = kmeans.labels_

# COMMAND ----------

rest_kmean_output = df_embs_original.copy()
rest_kmean_output["cluster"] = labels

# COMMAND ----------

rest_kmean_output

# COMMAND ----------

rest_kmean_output[rest_kmean_output["cluster"] == i].groupby("Y").size()

# COMMAND ----------

for i in range(0,(k)):
    plt.figure()  # create a new figure for each chart
    rest_kmean_output[rest_kmean_output["cluster"] == i].groupby("Y").size().plot.bar()
    plt.title(f"Cluster {i}")  # add a title to the chart

# COMMAND ----------

for k in range(50,100,10):
    print(k)
    kmeans = KMeans(n_clusters=k)
    X = df_embs_original.drop(["Y","atg_code"],axis = 1)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    score = silhouette_score(X, labels)

    # Print the R-squared statistic
    print("silhouette_score:", score)

# COMMAND ----------

