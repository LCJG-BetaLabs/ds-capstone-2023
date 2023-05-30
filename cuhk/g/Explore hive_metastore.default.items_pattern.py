# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT * FROM `hive_metastore`.`default`.`items_pattern`;

# COMMAND ----------

_sqldf.toPandas()

# COMMAND ----------

