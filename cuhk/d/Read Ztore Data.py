# Databricks notebook source
import os
container = "data2"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.parquet(os.path.join(data_path, "Ztore-data", "project-data-20230316"))


# COMMAND ----------

df1 = spark.read.parquet(os.path.join(data_path, "Ztore-data", "20230321-training-queries.parquet"))
display(df1)

# COMMAND ----------

import pandas as pd
team_container = "capstone2023-cuhk-team-d"
team_path = f"abfss://{team_container}@capstone2023cuhk.dfs.core.windows.net/"


# COMMAND ----------

files = dbutils.fs.ls(os.path.join(team_path))
for i in files:
    print(i)


# COMMAND ----------

display(df)

# COMMAND ----------

query_df = spark.sql("SELECT * FROM df")

# COMMAND ----------

