# Databricks notebook source
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

# COMMAND ----------

container ="data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv"), header=True)
df2 = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv"), header=True)
df = df.toPandas()
df2 = df2.toPandas()
df = df.loc[df['PRICE']!='0']
df = df.loc[df['QTY_SALES']!='0']
df2 = df2.loc[df2['PRICE']!='0']
df2 = df2.loc[df2['QTY_SALES']!='0']
# df.loc[df['QTY_SALES']=='MBI       ']
df = df.drop(index=[583])

# COMMAND ----------

df['COST'] = pd.to_numeric(df['COST'], errors='coerce')
df = df.dropna(subset=['COST'])
df['COST'] = df['COST'].astype(float)
df['PRICE'] = df['PRICE'].astype(float)
df['QTY_SALES'] = df['QTY_SALES'].astype(float)

# COMMAND ----------

df2['COST'] = pd.to_numeric(df2['COST'], errors='coerce')
df2 = df2.dropna(subset=['COST'])
df2['COST'] = df2['COST'].astype(float)
df2['PRICE'] = df2['PRICE'].astype(float)
df2['QTY_SALES'] = df2['QTY_SALES'].astype(float)

# COMMAND ----------

df['QTY_SALES'].describe()

# COMMAND ----------

df2['QTY_SALES'].describe()

# COMMAND ----------

df2['PUBLISHER'].str.strip().unique()

# COMMAND ----------

df['PUBLISHER'].str.strip().unique()

# COMMAND ----------

temp = set(df['PUBLISHER'].str.strip().unique()) & set(df2['PUBLISHER'].str.strip().unique())

# COMMAND ----------

temp

# COMMAND ----------

df2 = df2[df2['VENDOR'].str.strip().isin(set(df['VENDOR'].str.strip().unique()) & set(df2['VENDOR'].str.strip().unique()))]
df2 = df2[df2['BRAND'].str.strip().isin(set(df['BRAND'].str.strip().unique()) & set(df2['BRAND'].str.strip().unique()))]
df2 = df2[df2['PRD_CATEGORY'].str.strip().isin(set(df['PRD_CATEGORY'].str.strip().unique()) & set(df2['PRD_CATEGORY'].str.strip().unique()))]
df2 = df2[df2['PRD_ORIGIN'].str.strip().isin(set(df['PRD_ORIGIN'].str.strip().unique()) & set(df2['PRD_ORIGIN'].str.strip().unique()))]
df2 = df2[df2['PUBLISHER'].str.strip().isin(set(df['PUBLISHER'].str.strip().unique()) & set(df2['PUBLISHER'].str.strip().unique()))]

# COMMAND ----------



# COMMAND ----------

df2

# COMMAND ----------

PUBLISHERList = set(df['PUBLISHER'].str.strip().unique()) & set(df2['PUBLISHER'].str.strip().unique())
df = df[df['PUBLISHER'].str.strip().isin(PUBLISHERList)]
df2 = df2[df2['PUBLISHER'].str.strip().isin(PUBLISHERList)]

# COMMAND ----------

len(set(df['PUBLISHER'].str.strip().unique()) &set(df2['PUBLISHER'].str.strip().unique()))

# COMMAND ----------

len(df['PUBLISHER'].str.strip().unique()) , len(df2['PUBLISHER'].str.strip().unique())

# COMMAND ----------

df['QTY_SALES'].plot.box()

# COMMAND ----------

df["QTY_SALES"].describe(include="all")

# COMMAND ----------

df.loc[df['QTY_SALES'] < 40]

# COMMAND ----------

from sklearn.preprocessing import minmax_scale
df["STD_SALES"] = minmax_scale(df["QTY_SALES"], feature_range=(0, 1), axis=0, copy=True)
df2["STD_SALES"] = minmax_scale(df2["QTY_SALES"], feature_range=(0, 1), axis=0, copy=True)

# COMMAND ----------

df["STD_SALES"].describe()

# COMMAND ----------

df["STD_SALES"].plot.box()

# COMMAND ----------

df["STD_SALES"].quantile(0.85)

# COMMAND ----------

df_train.loc[df_train['STD_SALES'] < df_train["STD_SALES"].quantile(0.85)]