# Databricks notebook source
import os
import pandas as pd

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df1 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv")).toPandas()

df2 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv")).toPandas()

df3 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_2021_2+BOOKS.csv")).toPandas()

df4 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_2022_2+BOOKS.csv")).toPandas()

trans_df = pd.concat([df3, df4], ignore_index=True)
item_df = pd.concat([df1, df2], ignore_index=True)
print(item_df.shape, trans_df.shape)