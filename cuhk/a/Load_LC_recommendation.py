# Databricks notebook source
import os

# LC
container = "data3" 
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df2 = spark.read.format("csv").load(os.path.join(data_path, "pattern_recognition", "attribute.csv"))
df2.show()