# Databricks notebook source
pip install tensorflow==2.10

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
# using pyspark
container = "data3"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.format("csv").load(os.path.join(data_path,"pattern_recognition", "attribute.csv"))
df.display()

# COMMAND ----------


# using pandas
import pandas as pd
dbutils.fs.cp(os.path.join(data_path, "pattern_recognition"),
"file:/pattern_recognition", recurse=True) # copy folder from ABFS to local
# Load data
data = pd.read_csv(os.path.join("file:/pattern_recognition", "attribute.csv"))
data.head()
# To list out file/dirs
dbutils.fs.ls("file:/pattern_recognition")

# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
# Define data
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
# Create a PySpark DataFrame
df = spark.createDataFrame(data, ["Name", "Age"])
# Write parquet from storage
df.write.mode('overwrite').parquet(os.path.join(team_path, "test_parquet"))
# Read parquet from storage
read_df = spark.read.parquet(os.path.join(team_path, "test_parquet"))
display(read_df)
# Saving model

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


# COMMAND ----------

model.save("file:/databricks/driver/file:/tmp/test_model")
dbutils.fs.cp("file:/databricks/driver/file:/tmp/test_model", os.path.join(team_path, "test_model"),recurse=True) # copy folder from local to ABFS
# load model
dbutils.fs.cp(os.path.join(team_path, "test_model"), "file:/databricks/driver/file:/tmp/test_model",recurse=True) # copy folder from ABFS to local
model = tf.keras.models.load_model("file:/databricks/driver/file:/tmp/test_model")

# COMMAND ----------

display(dbutils.fs.ls("file:/databricks/driver/file:/tmp"))

# COMMAND ----------

# MAGIC %fs
# MAGIC ls 