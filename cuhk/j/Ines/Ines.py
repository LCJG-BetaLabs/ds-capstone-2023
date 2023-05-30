# Databricks notebook source
# MAGIC %md
# MAGIC ### Read Dataset

# COMMAND ----------

import os

# Data path (read-only)

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df1 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv"), header=True) 
df_items2021 = df1.toPandas()
df2 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv"), header=True) 
df_items2022 = df2.toPandas()
df3 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_2021_2+BOOKS.csv"), header=True) 
df_trans2021 = df3.toPandas()
df4 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_2022_2+BOOKS.csv"), header=True)
df_trans2022 = df4.toPandas()

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import requests
from time import sleep
from datetime import date, timedelta, datetime
from urllib.request import Request, urlopen
import os
import time
import datetime
import numpy as np
import re
import urllib.request
import base64
import json
from urllib.parse import urlencode
import matplotlib.pyplot as plt

# COMMAND ----------

df_items_raw = pd.concat([df_items2021,df_items2022])
df_items_raw.head()

# COMMAND ----------

df_items_raw.info()

# COMMAND ----------

df_tran_raw = pd.concat([df_trans2021,df_trans2022])
df_tran_raw.head()

# COMMAND ----------

df_tran_raw.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Pre-processing & Data Cleansing

# COMMAND ----------

# DBTITLE 1,1. Replacing "NULL", empty or meaningless values to np.nan
df_items_raw = df_items_raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_items_raw = df_items_raw.replace({'NULL':np.nan, ' ':np.nan, '':np.nan, '"':np.nan, None:np.nan})
df_items_raw.sort_values(['PRODUCT_ID','TITLE'])

# COMMAND ----------

# DBTITLE 0,Check null values
df_items_raw.info()

# COMMAND ----------

# DBTITLE 1,2. Drop NULL product ID rows
df_items_raw_1 = df_items_raw[df_items_raw['PRODUCT_ID'].notnull()].sort_values('PRODUCT_ID')

df_items_raw_1.head()

# COMMAND ----------

df_items_raw_1.info()

# COMMAND ----------

# DBTITLE 1,3. Drop NULL cost rows
df_items_raw_2 = df_items_raw_1[df_items_raw_1['COST'].notnull()].sort_values('ISBN13',ascending=False)
df_items_raw_2.head()

# COMMAND ----------

df_items_raw_2.info()

# COMMAND ----------

# DBTITLE 1,4. Fix data shifted row
df_items_raw_2_trim = pd.DataFrame([df_items_raw_2.sort_values('ISBN13',ascending=False).iloc[0]])
df_items_raw_2_trim['TITLE'] = df_items_raw_2_trim['TITLE']+', '+df_items_raw_2_trim['COST']
df_items_raw_2_trim['TITLE'] = df_items_raw_2_trim['TITLE'].str.strip('"')
df_items_raw_2_trim['TITLE'] = df_items_raw_2_trim['TITLE'].str.replace('""','"')
df_items_raw_2_trim

# COMMAND ----------

df_items_raw_3_trim = pd.concat([df_items_raw_2_trim.iloc[:, 0:3],df_items_raw_2_trim.iloc[:, 3:].shift(periods=-1, axis="columns")],axis=1)
df_items_raw_3_trim

# COMMAND ----------

df_items_raw_3 = pd.concat([df_items_raw_3_trim,df_items_raw_2.iloc[1:]])
df_items_raw_3.head()

# COMMAND ----------

df_items_raw_3.info()

# COMMAND ----------

df_items = df_items_raw_3.sort_values(['CREATE_DATE','ISBN13'])
df_items['COST'] = pd.to_numeric(df_items['COST'])
df_items['PRICE'] = pd.to_numeric(df_items['PRICE'])
df_items['QTY_PURCHASE'] = pd.to_numeric(df_items['QTY_PURCHASE'])
df_items['QTY_SALES'] = pd.to_numeric(df_items['QTY_SALES'])
df_items['QTY_STOCK'] = pd.to_numeric(df_items['QTY_STOCK'])
df_items['CREATE_DATE'] = pd.to_datetime(df_items['CREATE_DATE']).dt.strftime('%Y-%m-%d')
df_items['BOOK_ORGPR'] = df_items['BOOK_ORGPR'].str.strip('$')
df_items['BOOK_ORGPR'] = pd.to_numeric(df_items['BOOK_ORGPR'])
df_items['BOOK_DATE'] = pd.to_datetime(df_items['BOOK_DATE']).dt.strftime('%Y-%m-%d')
df_items['BOOK_PAGES'] = pd.to_numeric(df_items['BOOK_PAGES'])
df_items.head()

# COMMAND ----------

df_items[df_items['BOOK_ORGPR']=='PRODUCT']

# COMMAND ----------

df_items.dtypes