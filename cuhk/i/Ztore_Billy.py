# Databricks notebook source
# MAGIC %md
# MAGIC # Core Part

# COMMAND ----------

# KMeans
# import librabries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# config
num_cluster = 9

# import data
data_df = pd.read_csv('query_dummy_variables.csv')
binary_columns = data_df.values.transpose() # convert to np array

col_header = data_df.columns

# Cluster the binary columns using k-means with 4 clusters
kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(binary_columns)

# Group the column names based on their k-means cluster assignments
column_groups = [[] for _ in range(num_cluster)]
for i, label in enumerate(kmeans.labels_):
    column_name = col_header[i]
    column_groups[label].append(column_name)

# Combine the column groups into a list of strings
result = []
for group in column_groups:
    result.append(" & ".join(group))

for n, _ in enumerate(result):
    print(f'idx {n}: {_}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Messy stuffs

# COMMAND ----------

import os

# Data path (read-only)

# ZStore
container = "data2"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.parquet(os.path.join(data_path, "Ztore-data", "project-data-20230316"))

display(df)

# COMMAND ----------

df.createOrReplaceTempView("Ztore")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM Ztore;

# COMMAND ----------

df = spark.sql(f'''
SELECT * FROM Ztore
''')
type(df)