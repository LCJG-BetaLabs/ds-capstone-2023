# Databricks notebook source
# MAGIC %md
# MAGIC # Import

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

spark.sparkContext

# COMMAND ----------

# MAGIC %md
# MAGIC # Schema

# COMMAND ----------

t_raw.show()

# COMMAND ----------

t_raw.printSchema()

# COMMAND ----------

q_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Database Transformation

# COMMAND ----------

# DBTITLE 1,Column under struct<list:array<struct<item:string>>> structure
# for instance active_1_district
#  |-- active_1_district: struct (nullable = true)
#  |    |-- list: array (nullable = true)
#  |    |    |-- element: struct (containsNull = true)
#  |    |    |    |-- item: string (nullable = true)
list_col = [
    'active_1_district',
    'active_2_district',
    'category_11', 
    'category_12',
    'category_13',
    'category_21',
    'category_22',
    'category_31',
    'category_41',
    'category_42',
]

# COMMAND ----------

# DBTITLE 1,Unnest struct<list:array<struct<item:string>>> structure into array
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

# DBTITLE 1,Show number of items in column with array
t.select([size(i).alias(i) for i in list_col]).select([max(i).alias(i) for i in list_col]).show()

# COMMAND ----------

# DBTITLE 1,Unnest array into separate column
# max number of items in each column
dict_col = {
    'active_1_district': 3,
    'active_2_district': 3,
    'category_11': 12, 
    'category_12': 12,
    'category_13': 12,
    'category_21': 8,
    'category_22': 8,
    'category_31': 18,
    'category_41': 6,
    'category_42': 6,
}
# expand struct<list:array<struct<item:string>>> structure into separate column
list_col = []
for key, value in dict_col.items():
    for v in range(value):
        list_col.append((col(key)[v]).alias(key+'_'+str(v+1))) 
# unnest struct<list:array<struct<item:string>>> structure
t = t.select(['user_id', 'gender', 'age_range', 'date'] + list_col)

# COMMAND ----------

# DBTITLE 1,Distinct "gender" value
list_gender = [g['gender'] for g in t.select('gender').distinct().collect()]
list_gender = [g for g in list_gender if g != None]
list_gender = set(list_gender)
list_gender

# COMMAND ----------

# DBTITLE 1,Distinct "age_range" value
list_age_range = [a['age_range'] for a in t.select('age_range').distinct().collect()]
list_age_range = [a for a in list_age_range if a != None]
list_age_range = set(list_age_range)
list_age_range

# COMMAND ----------

# DBTITLE 1,Distinct "district" value
list_district = []
for district in ['active_1_district_1', 'active_1_district_2', 'active_1_district_3', 'active_2_district_1', 'active_2_district_2', 'active_2_district_3']:
    list_district.extend([d[district] for d in t.select(district).distinct().collect()])
list_district = [d for d in list_district if d != None]
list_district = set(list_district)
list_district

# COMMAND ----------

# DBTITLE 1,Modified column for "gender"
for g in list_gender:
    t = t.withColumn(g, F.when(F.col("gender").isin(g), 1).otherwise(0))

# COMMAND ----------

# DBTITLE 1,Modified column for "age_range"
for a in list_age_range:
    t = t.withColumn(a, F.when(F.col("age_range").isin(a), 1).otherwise(0))

# COMMAND ----------

# DBTITLE 1,Modified column for "district"
for d in list_district:
    t = t.withColumn(d, F.when(F.col("active_1_district_1").isin(d), 1).otherwise(0))
    t = t.withColumn(d, F.when(F.col("active_1_district_2").isin(d), 1).otherwise(F.col(d)))

# COMMAND ----------

# DBTITLE 1,Modified column for "period"
t = t.withColumn('datediff', F.expr('datediff(max(date) over(), date)'))
t = t.withColumn('10 Days', F.when(F.col('datediff') <= 10, 1).otherwise(0))
t = t.withColumn('30 Days', F.when(F.col('datediff') <= 30, 1).otherwise(0))
t = t.withColumn('90 Days', F.when(F.col('datediff') <= 90, 1).otherwise(0))
t = t.withColumn('All time', F.when(F.col('date').isNotNull(), 1).otherwise(0))

# COMMAND ----------

# DBTITLE 1,List out all required columns
list_col = []
# get gender
list_col.extend(sorted(set(list_gender)))
# get age_range
list_col.extend(sorted(set(list_age_range)))
# get district
list_col.extend(sorted(set(list_district)))
# max number of items in each column
dict_col = {
    'category_11': 12, 
    'category_12': 12,
    'category_13': 12,
    'category_21': 8,
    'category_22': 8,
    'category_31': 18,
    'category_41': 6,
    'category_42': 6,
}
# expand struct<list:array<struct<item:string>>> structure into separate column
for key, value in dict_col.items():
    for v in range(value):
        list_col.append(key+'_'+str(v+1))
list_col 

# COMMAND ----------

t = t.select(['user_id', '10 Days', '30 Days', '90 Days'] + list_col)
t.createOrReplaceTempView("t")

# COMMAND ----------

# DBTITLE 1,Save the result
# t.write.mode("overwrite").parquet("dbfs:/t.parquet")
t = spark.read.parquet("dbfs:/t.parquet")
t.createOrReplaceTempView("t")

# COMMAND ----------

spark.sql('SELECT * FROM t').show(1)

# COMMAND ----------

