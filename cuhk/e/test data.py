# Databricks notebook source
import os
import pandas as pd

# COMMAND ----------

container ="data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df_2021 = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv"), header=True)
df_2022 = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv"), header=True)
df_2021 = df_2021.toPandas()
df_2022 = df_2022.toPandas()

# COMMAND ----------

df_2021

# COMMAND ----------

df_2022

# COMMAND ----------

df_2022.describe()

# COMMAND ----------

df_2021.groupby(["PRODUCT_ID"]).sum()

# COMMAND ----------

