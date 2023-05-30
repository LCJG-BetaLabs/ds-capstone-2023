# Databricks notebook source
import os
import pandas as pd
import csv

# LC
container = "data3" 
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.format("csv").option("header", "true").load(os.path.join(data_path, "pattern_recognition", "attribute.csv"))
display(df)


# COMMAND ----------

pwd