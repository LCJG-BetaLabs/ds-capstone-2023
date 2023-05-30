# Databricks notebook source
# MAGIC %md
# MAGIC # This script is to understand and explore the transaction dataset

# COMMAND ----------

import os
from pathlib import Path
import pyspark.sql.functions as F
from pyspark.sql.functions import size, expr, col, max, explode, when, udf, array_contains
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

# COMMAND ----------

# unnest struct<list:array<struct<item:string>>> structure
t = t_raw.selectExpr(
    'user_id',
    'gender',
    'age_range',
    'date',
    '(transform(active_1_district.*, x -> x.item)) as active_1_district',
    '(transform(active_2_district.*, x -> x.item)) as active_2_district',
    '(transform(category_engagement_11.*, x -> x.item)) as category_11',
    '(transform(category_engagement_12.*, x -> x.item)) as category_12',
    '(transform(category_engagement_13.*, x -> x.item)) as category_13',
    '(transform(category_engagement_21.*, x -> x.item)) as category_21',
    '(transform(category_engagement_22.*, x -> x.item)) as category_22',
    '(transform(category_engagement_31.*, x -> x.item)) as category_31',
    '(transform(category_engagement_41.*, x -> x.item)) as category_41',
    '(transform(category_engagement_42.*, x -> x.item)) as category_42',
 )

# COMMAND ----------

t = t.withColumn('datediff', F.expr('datediff(max(date) over(), date)'))
t = t.withColumn('period', F.array(
    F.when(F.col('datediff') <= 10, "10 Days"),
    F.when(F.col('datediff') <= 30, "30 Days"),
    F.when(F.col('datediff') <= 90, "90 Days"),
    F.when(F.col('date').isNotNull(), "All time"),
))

# COMMAND ----------

t = t.select([
    'user_id', 'gender', 'age_range', 'period', 
    'active_1_district', 'active_2_district', 
    'category_11', 'category_12', 'category_13',
    'category_21', 'category_22',
    'category_31',
    'category_41', 'category_42',
])
t.createOrReplaceTempView("transaction")

# COMMAND ----------

queries = """
    SELECT
        COUNT(DISTINCT user_id), 
        gender, period,
        category_21, category_11
    FROM 
        transaction
    WHERE
        age_range IN ('Age Range 1', 'Age Range 5', 'Age Range 4', 'Age Range 3') and 
        (arrays_overlap(active_1_district, array('District 2', 'District 3')) or 
        arrays_overlap(active_2_district, array('District 2', 'District 3')))     
    GROUP BY
        gender, period, 
        category_21, category_11
"""

# COMMAND ----------

spark.sql(queries).show()