# Databricks notebook source
import os

# Data path (read-only)

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df1 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv")) 

df2 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv")) 

df3 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_2021_2+BOOKS.csv")) 

df4 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_2022_2+BOOKS.csv"))

display(df1)
display(df2)
display(df3)
display(df4)

# COMMAND ----------

display(df1)

# COMMAND ----------

display(df2)

# COMMAND ----------

display(df3)

# COMMAND ----------

display(df4)