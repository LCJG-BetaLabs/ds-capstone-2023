# Databricks notebook source
# MAGIC %md
# MAGIC # This script is to transform the query table into binary table format for training clustering algorithms later

# COMMAND ----------

# MAGIC %md
# MAGIC # Import

# COMMAND ----------

import os
import pandas as pd
from pathlib import Path
import pyspark.sql.functions as F
from pyspark.sql.functions import size, expr, col, max, explode, when, udf, array_contains, collect_set
# Data path (read-only)
# ZStore transaction
container = "data2"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
t_raw = spark.read.parquet(os.path.join(data_path, "Ztore-data", "project-data-20230316"))
t_raw.createOrReplaceTempView("t_raw")
# ZStore query
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/" 
q_raw = spark.read.parquet(os.path.join(data_path, "Ztore-data", "20230321-training-queries.parquet"))
q_raw.createOrReplaceTempView("q_raw")
# Transformed transaction
t = spark.read.parquet("dbfs:/t.parquet")
t.createOrReplaceTempView("t")

# COMMAND ----------

list_where = sorted(set([i['WHERE_EXPLODE'] for i in q_raw.withColumn("WHERE_EXPLODE", explode(col("WHERE"))).select("WHERE_EXPLODE").distinct().collect()]))
list_select = sorted(set([i['SELECT_EXPLODE'] for i in q_raw.witÍÍhColumn("SELECT_EXPLODE", explode(col("SELECT"))).select("SELECT_EXPLODE").distinct().collect()]))
list_groupby = sorted(set([i['GROUPBY_EXPLODE'] for i in q_raw.withColumn("GROUPBY_EXPLODE", explode(col("GROUP BY"))).select("GROUPBY_EXPLODE").distinct().collect()]))

# COMMAND ----------

for i in list_where:
    q_raw = q_raw.withColumn(i, F.when(array_contains(F.col("WHERE"), i), 1).otherwise(0))

# COMMAND ----------

q_raw.select(list_where).write.options(header=True, delimiter=',').mode("overwrite").csv("dbfs:/queries.csv")

# COMMAND ----------

q = spark.read.options(header=True, delimiter=',').csv("dbfs:/queries.csv")
display(q.toPandas())

# COMMAND ----------

