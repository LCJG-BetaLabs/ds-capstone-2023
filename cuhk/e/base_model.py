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

df3 = df[['COST','PRICE', 'QTY_SALES','VENDOR','BRAND','PRD_CATEGORY','PRD_ORIGIN','PUBLISHER']]
df3

# COMMAND ----------

df3['COST'] = pd.to_numeric(df['COST'], errors='coerce')
df3 = df3.dropna(subset=['COST'])
df3['COST'] = df3['COST'].astype(float)
df3['PRICE'] = df3['PRICE'].astype(float)
df3['QTY_SALES'] = df3['QTY_SALES'].astype(float)

# COMMAND ----------

df3 = pd.get_dummies(df3, prefix="cat", 
                            columns=["VENDOR","BRAND","PRD_CATEGORY","PRD_ORIGIN","PUBLISHER"], 
                            drop_first=False)

# COMMAND ----------

#define trainging data set and testing data set
training_data, testing_data = train_test_split(df3, test_size=0.2, random_state=25)

# COMMAND ----------

x_data = training_data.drop(labels='QTY_SALES',axis=1)
y_data = training_data['QTY_SALES']
x_data_test = testing_data.drop(labels='QTY_SALES',axis=1)
y_data_test = testing_data['QTY_SALES']

# COMMAND ----------

all_features = keras.Input(shape=x_data.shape[1])
x1 = layers.Dense(512, activation="relu")(all_features)# add more layer and notes
x2 = layers.Dropout(0.3)(x1)#change drop out rate to 0.3
x3 = layers.Dense(512, activation="relu")(x2) # add more layer and notes
x4 = layers.Dropout(0.3)(x3)#change drop out rate to 0.3
x5 = layers.Dense(512, activation="relu")(x4) # add more layer and notes
x6 = layers.Dropout(0.3)(x5)#change drop out rate to 0.3
output = layers.Dense(1)(x6)
model = keras.Model(all_features, output)
model.compile(loss='mse',
              optimizer=Adam(),
              metrics=['mse'])

# COMMAND ----------

model.fit(x=x_data, y=y_data, epochs=150)

# COMMAND ----------

out = model.predict(x_data_test)

# COMMAND ----------

def accuracy(y_data_test, predicted_y, threshold):
    result = []
    for x,y in zip(y_data_test.values.tolist() , predicted_y):
#       print(x , y[0] , x*(1-threshold) , x*(1+threshold) , x*x*(1-threshold) <= y[0] <= x*(1+threshold))
      result.append(x*(1-threshold) <= y[0] <= x*(1+threshold))
    i=0
    for x in result:
        if x==True:
            i = i+1
    print(len(result), i , i/len(result))

# COMMAND ----------

accuracy(y_data_test, out,0.5)

# COMMAND ----------

df4 = df2[['COST','PRICE', 'QTY_SALES','VENDOR','BRAND','PRD_CATEGORY','PRD_ORIGIN','PUBLISHER']]
df4['COST'] = pd.to_numeric(df2['COST'], errors='coerce')
df4 = df4.dropna(subset=['COST'])
df4['COST'] = df4['COST'].astype(float)
df4['PRICE'] = df4['PRICE'].astype(float)
df4['QTY_SALES'] = df4['QTY_SALES'].astype(float)
df4 = pd.get_dummies(df4, prefix="cat", 
                            columns=["VENDOR","BRAND","PRD_CATEGORY","PRD_ORIGIN","PUBLISHER"], 
                            drop_first=False)
df4 = df4.drop(labels='QTY_SALES',axis=1)


# COMMAND ----------

out2022 = model.predict(df4)

# COMMAND ----------

accuracy(df2['QTY_SALES'].astype(float), out2022,0.8)

# COMMAND ----------

model.save('/dbfs/FileStore/ann_model.h150')

# COMMAND ----------



# COMMAND ----------

display(dbutils.fs.ls("/FileStore"))

# COMMAND ----------

ann = keras.models.load_model("/dbfs/FileStore/ann_model.h150")

# COMMAND ----------


# check xgboost version
import xgboost as xg
print(xgboost.__version__)

# COMMAND ----------

XGmodel = xg.XGBRegressor()

# COMMAND ----------

XGmodel.fit(x_data, y_data)

# COMMAND ----------

XGout = XGmodel.predict(x_data_test)

# COMMAND ----------

def XGaccuracy(y_data_test, predicted_y, threshold):
    result = []
    for x,y in zip(y_data_test.values.tolist() , predicted_y):
#       print(x , y[0] , x*(1-threshold) , x*(1+threshold) , x*x*(1-threshold) <= y[0] <= x*(1+threshold))
      result.append(x*(1-threshold) <= y <= x*(1+threshold))
    i=0
    for x in result:
        if x==True:
            i = i+1
    print(len(result), i , i/len(result))

# COMMAND ----------

XGaccuracy(y_data_test, XGout,0.5)

# COMMAND ----------

XGout2022 = XGmodel.predict(df4)

# COMMAND ----------

XGaccuracy(df2['QTY_SALES'].astype(float), XGout,0.8)

# COMMAND ----------

XGmodel.save_model('/dbfs/FileStore/XGmodel.json')

# COMMAND ----------

model2 = xg.XGBRegressor()
model2.load_model("/dbfs/FileStore/XGmodel.json")