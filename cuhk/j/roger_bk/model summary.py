# Databricks notebook source
# MAGIC %md # model summary

# COMMAND ----------

!ls /dbfs/

# COMMAND ----------

!ls /dbfs/team_j

# COMMAND ----------

# from tensorflow.keras.utils import plot_model
import mlflow.keras
import os
import matplotlib.pyplot as plt


# COMMAND ----------

# MAGIC %md # image (VGG16) + text (preprocessed by BertTokenizer) concatenated binary classifier 

# COMMAND ----------

model_uri = 'runs:/5b5cda147afb42388d4aaeea222b46c1/model'
loaded_model = mlflow.keras.load_model(model_uri)

# COMMAND ----------

print(loaded_model.summary())

# COMMAND ----------

from tensorflow import keras
import matplotlib.pyplot as plt
# Load the saved Keras model from a directory
model_dir = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426'
loaded_model = keras.models.load_model(model_dir)

# COMMAND ----------

dir(loaded_model)

# COMMAND ----------

loaded_model.summary()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

