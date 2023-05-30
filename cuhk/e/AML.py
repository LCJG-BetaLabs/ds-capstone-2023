# Databricks notebook source
import os
import pandas as pd

# COMMAND ----------

container ="data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv"), header=True)
#df = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv"), header=True)
df = df.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

#need remove invalid value
df.loc[df['QTY_SALES']=='MBI       ']

# COMMAND ----------

df = df.drop(index=[583])

# COMMAND ----------

df['SALES'] = df['PRICE'].astype(float) * df['QTY_SALES'].astype(float)

# COMMAND ----------

df['PRD_CATEGORY'].str.strip().unique()

# COMMAND ----------

container = "data1" 
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/" 

df3 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_2021_2+BOOKS.csv")) 
df3 = df3.toPandas()

# COMMAND ----------

df3.head()

# COMMAND ----------

# If you want to use pandas
import pandas as pd
dbutils.fs.cp(os.path.join(data_path, "competitor_analysis"), "file:/competitor_analysis", recurse=True) # copy folder from ABFS to local

# Load data
data = pd.read_csv(os.path.join("file:/competitor_analysis", "attribute.csv"))
data.head()

# To list out file/dirs
dbutils.fs.ls("file:/competitor_analysis")

# Storage path for teams
team_container = "capstone2023-cuhk-team-b"
team_path = f"abfss://{team_container}@capstone2023cuhk.dfs.core.windows.net/"

# Define data
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]

# Create a PySpark DataFrame
df = spark.createDataFrame(data, ["Name", "Age"])

# Write parquet from storage
df.write.mode('overwirte').parquet(os.path.join(team_path, "test_parquet"))

# Read parquet from storage
read_df = spark.read.parquet(path)
display(read_df)

# Saving model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
x_train = np.random.rand(100, 3)
y_train = np.random.randint(2, size=100)
all_features = tf.keras.Input(shape = x_train.shape[1])
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(all_features, output)
model.compile(loss="binary_crossentropy", 
             optimizer=Adam(),
             metrics=["accuracy"])

model.fit(x=x_train, y=y_train, epochs=10)
data = model.save("file:/tmp/test_model")
dbutils.fs.cp("file:/tmp/test_model", os.path.join(team_path, "test_model"), recurse=True) # copy folder from local to ABFS

# load model
dbutils.fs.cp(os.path.join(team_path, "test_model"), "file:/tmp/test_model", recurse=True) # copy folder from ABFS to local
model = tf.keras.models.load_model("file:/tmp/test_model")