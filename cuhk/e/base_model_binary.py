# Databricks notebook source
import os
import tensorflow as tf
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
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter
import visualkeras
from PIL import ImageFont

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

VENDORList = set(df['VENDOR'].str.strip().unique()) & set(df2['VENDOR'].str.strip().unique())
df = df[df['VENDOR'].str.strip().isin(VENDORList)]
df2 = df2[df2['VENDOR'].str.strip().isin(VENDORList)]

# COMMAND ----------

BRANDList = set(df['BRAND'].str.strip().unique()) & set(df2['BRAND'].str.strip().unique())
df = df[df['BRAND'].str.strip().isin(BRANDList)]
df2 = df2[df2['BRAND'].str.strip().isin(BRANDList)]

# COMMAND ----------

PRD_CATEGORYList = set(df['PRD_CATEGORY'].str.strip().unique()) & set(df2['PRD_CATEGORY'].str.strip().unique())
df = df[df['PRD_CATEGORY'].str.strip().isin(PRD_CATEGORYList)]
df2 = df2[df2['PRD_CATEGORY'].str.strip().isin(PRD_CATEGORYList)]

# COMMAND ----------

PRD_ORIGINList = set(df['PRD_ORIGIN'].str.strip().unique()) & set(df2['PRD_ORIGIN'].str.strip().unique())
df = df[df['PRD_ORIGIN'].str.strip().isin(PRD_ORIGINList)]
df2 = df2[df2['PRD_ORIGIN'].str.strip().isin(PRD_ORIGINList)]

# COMMAND ----------

PUBLISHERList = set(df['PUBLISHER'].str.strip().unique()) & set(df2['PUBLISHER'].str.strip().unique())
df = df[df['PUBLISHER'].str.strip().isin(PUBLISHERList)]
df2 = df2[df2['PUBLISHER'].str.strip().isin(PUBLISHERList)]

# COMMAND ----------

df2 = df2.loc[df2['BRAND']!='S-ZONE  ']
df2 = df2.loc[df2['VENDOR']!='HINKLER   ']

# COMMAND ----------

def categorizeSales(threshold, n):
    return 1.0 if n >= threshold else 0.0

# COMMAND ----------

df3 = df[['COST','PRICE', 'QTY_SALES','VENDOR','BRAND','PRD_CATEGORY','PRD_ORIGIN','PUBLISHER']]
df3

# COMMAND ----------

df3['COST'] = pd.to_numeric(df['COST'], errors='coerce')
df3 = df3.dropna(subset=['COST'])
df3['COST'] = df3['COST'].astype(float)
df3['PRICE'] = df3['PRICE'].astype(float)
df3['QTY_SALES'] = df3['QTY_SALES'].astype(float)
df3["QTY_SALES"] = minmax_scale(df3["QTY_SALES"], feature_range=(0, 1), axis=0, copy=True)
df3 = pd.get_dummies(df3, prefix="cat", 
                            columns=["VENDOR","BRAND","PRD_CATEGORY","PRD_ORIGIN","PUBLISHER"], 
                            drop_first=False)

# COMMAND ----------

#define trainging data set and testing data set
training_data, testing_data = train_test_split(df3, test_size=0.2, random_state=25)

# COMMAND ----------

threshold = 0.3
x_data = training_data.drop(labels=['QTY_SALES'],axis=1)
y_data = training_data['QTY_SALES'].apply(lambda x: categorizeSales(threshold, x))
x_data_test = testing_data.drop(labels=['QTY_SALES'],axis=1)
y_data_test = testing_data['QTY_SALES'].apply(lambda x: categorizeSales(threshold, x))

# COMMAND ----------

x_data

# COMMAND ----------

model = keras.models.Sequential()
model.add(layers.Dense(512, activation='relu',input_dim=x_data.shape[1]))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])

# COMMAND ----------

image = visualkeras.layered_view(model, legend=True)

# COMMAND ----------

imgplot = plt.imshow(image)
plt.show()

# COMMAND ----------

model.summary()

# COMMAND ----------

len(x_data),len(y_data)

# COMMAND ----------

model.fit(x=x_data, y=y_data, epochs=100)

# COMMAND ----------

len(x_data_test) , len(y_data_test)

# COMMAND ----------

score, acc = model.evaluate(x_data_test, y_data_test)
print('Test score:', score)
print('Test accuracy:', acc)

# COMMAND ----------

df4 = df2[['COST','PRICE','VENDOR','BRAND','PRD_CATEGORY','PRD_ORIGIN','PUBLISHER','QTY_SALES']]
df4['COST'] = pd.to_numeric(df2['COST'], errors='coerce')
df4 = df4.dropna(subset=['COST'])
df4['COST'] = df4['COST'].astype(float)
df4['PRICE'] = df4['PRICE'].astype(float)
df4 = pd.get_dummies(df4, prefix="cat", columns=["VENDOR","BRAND","PRD_CATEGORY","PRD_ORIGIN","PUBLISHER"], drop_first=False)
df4['QTY_SALES'] = minmax_scale(df4["QTY_SALES"], feature_range=(0, 1), axis=0, copy=True)

# COMMAND ----------

y = df4['QTY_SALES'].apply(lambda x: categorizeSales(threshold, x))
x = df4.loc[:, ~df4.columns.isin(['QTY_SALES'])]

# COMMAND ----------

score, acc = model.evaluate(x, y)
print('Test score:', score)
print('Test accuracy:', acc)

# COMMAND ----------

y_pred = np.round(model.predict(x))
y_pred = y_pred.astype("float64")

# COMMAND ----------

np.nanmax(y_pred)

# COMMAND ----------

y

# COMMAND ----------

target_names = ['class 0', 'class 1']
print(classification_report(y, y_pred, target_names=target_names))

# COMMAND ----------

def runModel(df_train, df_test, model ,threshold,accuracy):
    x_train = df_train.drop(labels=['QTY_SALES'],axis=1)
    y_train = df_train["QTY_SALES"].apply(lambda x: categorizeSales(threshold, x))

    x_test = df_test.drop(labels=['QTY_SALES'],axis=1)
    y_test = df_test["QTY_SALES"].apply(lambda x: categorizeSales(threshold, x))

    model.fit(x=x_train, y=y_train, epochs=100,verbose=0)

    score, acc = model.evaluate(x_test, y_test)
    print("Threshold: {}".format(str(threshold)))
    print('Test score:', score)
    print('Test accuracy:', acc)
    target_names = ['class 0', 'class 1']

    y_train_pred = np.round(model.predict(x_train))
    y_train_pred = y_train_pred.astype("float64")
    print(classification_report(y_train, y_train_pred, target_names=target_names))

    y_pred = np.round(model.predict(x_test))
    y_pred = y_pred.astype("float64")
    print(classification_report(y_test, y_pred, target_names=target_names))

    accuracy= accuracy.append(acc)

# COMMAND ----------

accuracy = []
for t in range(1, 15):
    threshold = t/100
    runModel(df3, df4,model,threshold,accuracy)

# COMMAND ----------

df4["QTY_SALES"]

# COMMAND ----------

for t in range(1, 30):
    threshold = t/100
    temp = df3["QTY_SALES"].apply(lambda x: categorizeSales(threshold, x))
    print(threshold)
    print(Counter(temp))

# COMMAND ----------

df4["QTY_SALES"]

# COMMAND ----------

    temp = df4["QTY_SALES"].apply(lambda x: categorizeSales(0.05, x))
    print(Counter(temp))

# COMMAND ----------

accuracy