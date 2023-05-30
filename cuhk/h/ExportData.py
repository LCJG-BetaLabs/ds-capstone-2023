# Databricks notebook source
import os

# Data path (read-only)

container = "data3"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df1 = spark.read.format("csv").load(os.path.join(data_path, "competitor_analysis", "attribute.csv"), header=True) 

df2 = spark.read.format("csv").load(os.path.join(data_path, "pattern_recognition", "attribute.csv"), header=True) 

# COMMAND ----------

display(df1)

# COMMAND ----------

df1.describe()

# COMMAND ----------

print(data.iloc[2])

# COMMAND ----------

display(df2)

# COMMAND ----------

df1.union(df2)

# COMMAND ----------

display(df1)

# COMMAND ----------

