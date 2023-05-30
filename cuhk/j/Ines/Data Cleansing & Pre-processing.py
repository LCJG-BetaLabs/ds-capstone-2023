# Databricks notebook source
# MAGIC %md
# MAGIC #### Read Datasets

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
import matplotlib.pyplot as pltb

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Cleansing for 'ITEMS'

# COMMAND ----------

df_items_raw = pd.concat([df_items2021,df_items2022])
df_items_raw = df_items_raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_items_raw.iloc[583:584,] = df_items_raw.iloc[583:584,].shift(periods=2, axis=1)
df_items_raw.iloc[582:584,3:] = df_items_raw.iloc[582:584,3:].shift(periods=-1)
df_items_raw = df_items_raw.drop(df_items_raw.index[583]).reset_index().drop('index',axis=1)

df_items_raw.iloc[1043:1044,]['TITLE'] = df_items_raw.iloc[1043:1044,]['TITLE'] + ', ' + df_items_raw.iloc[1043:1044,]['COST']
df_items_raw.iloc[1043:1044,]['TITLE'] = df_items_raw.iloc[1043:1044,]['TITLE'].str.strip('"').str.replace('""','"')
df_items_raw.iloc[1043:1044,3:] = df_items_raw.iloc[1043:1044,3:].shift(-1,axis=1)

# COMMAND ----------

df_items_shift = df_items_raw[(df_items_raw['BOOK_ORGPR'].isnull())&(df_items_raw['COST'].notnull())]

df_items_shift['TRANSLATOR'] = np.where(df_items_shift['PRODUCT_ID']=='"',df_items_shift['CREATE_DATE'],df_items_shift['TRANSLATOR'])
df_items_shift['BOOK_ORGPR'] = np.where(df_items_shift['PRODUCT_ID']=='"',df_items_shift['TITLE'],df_items_shift['BOOK_ORGPR'])
df_items_shift['BOOK_DATE'] = np.where(df_items_shift['PRODUCT_ID']=='"',df_items_shift['COST'],df_items_shift['BOOK_DATE'])
df_items_shift['BOOK_PAGES'] = np.where(df_items_shift['PRODUCT_ID']=='"',df_items_shift['PRICE'],df_items_shift['BOOK_PAGES'])
df_items_shift['BOOK_COVER'] = np.where(df_items_shift['PRODUCT_ID']=='"',df_items_shift['QTY_PURCHASE'],df_items_shift['BOOK_COVER'])

df_items_shift['TRANSLATOR'] = df_items_shift['TRANSLATOR'].shift(-1)
df_items_shift['BOOK_ORGPR'] = df_items_shift['BOOK_ORGPR'].shift(-1)
df_items_shift['BOOK_DATE'] = df_items_shift['BOOK_DATE'].shift(-1)
df_items_shift['BOOK_PAGES'] = df_items_shift['BOOK_PAGES'].shift(-1)
df_items_shift['BOOK_COVER'] = df_items_shift['BOOK_COVER'].shift(-1)

df_items_shift = df_items_shift[df_items_shift['PRODUCT_ID']!='"']

# COMMAND ----------

df_items = pd.concat([df_items_raw,df_items_shift])
df_items = df_items[df_items['PRODUCT_ID']!='"']
df_items = df_items.drop_duplicates(['PRODUCT_ID'],keep='last')
df_items = df_items.replace({'NULL':np.nan, ' ':np.nan, '':np.nan, None:np.nan})
df_items = df_items.reset_index().drop('index',axis=1)
df_items['BOOK_ORGPR'] = np.where(df_items['BOOK_ORGPR']=='PRODUCT',np.nan,df_items['BOOK_ORGPR'])

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

df_items.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Cleansing for 'RECOMMENDATIONS'

# COMMAND ----------

df_trans2021 = df_trans2021.rename({'PRODUCT':'PRODUCT_ID'},axis=1)
df_recom_raw = pd.concat([df_trans2021,df_trans2022])
df_recom_raw = df_recom_raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_recom_raw = df_recom_raw.reset_index().reset_index().drop('index',axis=1)
# df_recom_raw['QUANTITY'] = pd.to_numeric(df_recom_raw['QUANTITY'])
# df_recom_raw[(df_recom_raw['PRODUCT']==df_recom_raw['PRODUCT_ID'])]

df_recom_raw

# COMMAND ----------

df_recom_shift = df_recom_raw[(df_recom_raw['SHOP_NO'].isnull())]

df_recom_shift['TITLE2'] = df_recom_shift['HASHED_INVOICE_ID']
df_recom_shift['TITLE2'] = np.where(~df_recom_shift['HASHED_INVOICE_ID'].str.startswith('0',na=False),df_recom_shift['HASHED_INVOICE_ID'],df_recom_shift['TITLE2'])
df_recom_shift['PRICE'] = np.where(~df_recom_shift['HASHED_INVOICE_ID'].str.startswith('0',na=False),df_recom_shift['TRANDATE'],df_recom_shift['PRICE'])
df_recom_shift['QUANTITY'] = np.where(~df_recom_shift['HASHED_INVOICE_ID'].str.startswith('0',na=False),df_recom_shift['PRODUCT_ID'],df_recom_shift['QUANTITY'])
df_recom_shift['AMOUNT'] = np.where(~df_recom_shift['HASHED_INVOICE_ID'].str.startswith('0',na=False),df_recom_shift['ISBN13'],df_recom_shift['AMOUNT'])
df_recom_shift['SHOP_NO'] = np.where(~df_recom_shift['HASHED_INVOICE_ID'].str.startswith('0',na=False),df_recom_shift['HASHED_CUSTOMER_ID'],df_recom_shift['SHOP_NO'])

df_recom_shift['TITLE2'] = df_recom_shift['TITLE2'].shift(-1)
df_recom_shift['PRICE'] = df_recom_shift['PRICE'].shift(-1)
df_recom_shift['QUANTITY'] = df_recom_shift['QUANTITY'].shift(-1)
df_recom_shift['AMOUNT'] = df_recom_shift['AMOUNT'].shift(-1)
df_recom_shift['SHOP_NO'] = df_recom_shift['SHOP_NO'].shift(-1)

df_recom_shift['TITLE'] = np.where(df_recom_shift['TITLE2']!='"',df_recom_shift['TITLE'] + ' ' + df_recom_shift['TITLE2'],df_recom_shift['TITLE'])
df_recom_shift = df_recom_shift.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)

df_recom_shift = df_recom_shift[df_recom_shift['HASHED_INVOICE_ID'].str.startswith('0', na=False)]

# df_recom_shift.sort_values('TITLE2').iloc[1130:1134,]
df_recom_shift

# COMMAND ----------

df_recom = pd.concat([df_recom_raw,df_recom_shift])
df_recom = df_recom.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_recom = df_recom[df_recom['HASHED_INVOICE_ID'].str.startswith('0', na=False)]
# df_recom = df_recom[df_recom['HASHED_INVOICE_ID']!='"']
# df_recom = df_recom.drop_duplicates(['HASHED_INVOICE_ID'],keep='last')
df_recom = df_recom.replace({'NULL':np.nan, ' ':np.nan, '':np.nan, None:np.nan})
df_recom = df_recom.dropna(subset=['QUANTITY'])

df_recom = df_recom.sort_values('level_0').reset_index().drop(['level_0','index','TITLE2'],axis=1)

df_recom = df_recom.sort_values('PRICE')
df_recom.iloc[-16:,6:] = df_recom.iloc[-16:,6:].shift(-1,axis=1)
df_recom.iloc[-11:-3,6:] = df_recom.iloc[-11:-3,6:].shift(-1,axis=1)
df_recom['AMOUNT'] = np.where(df_recom['AMOUNT'].isnull(),df_recom['PRICE'],df_recom['AMOUNT'])
df_recom = df_recom[df_recom['PRICE']!='Slowly'].sort_values(['HASHED_INVOICE_ID','SHOP_NO'])

df_recom['SHOP_NO'] = df_recom.groupby('HASHED_INVOICE_ID')['SHOP_NO'].ffill()

df_recom['TRANDATE'] = pd.to_datetime(df_recom['TRANDATE']).dt.strftime('%Y-%m-%d')
df_recom['PRICE'] = pd.to_numeric(df_recom['PRICE'])
df_recom['QUANTITY'] = pd.to_numeric(df_recom['QUANTITY'])
df_recom['AMOUNT'] = pd.to_numeric(df_recom['AMOUNT'])

pd.set_option('display.max_columns', None)  
# df_recom.iloc[-16:,5:]
# df_recom[df_recom['AMOUNT'].isnull()]
# df_recom[df_recom['SHOP_NO'].isnull()]
# df_recom[df_recom['HASHED_INVOICE_ID']=='0xBCA38D059D7D19B0623B15E60630C6F068539E32']
df_recom.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Join 'ITEMS' and 'RECOMMENDATIONS'

# COMMAND ----------

df_sales_group = df_recom.groupby(['HASHED_INVOICE_ID','TRANDATE','PRODUCT_ID','HASHED_CUSTOMER_ID','SHOP_NO','PRICE'])['QUANTITY','AMOUNT'].sum().reset_index()

df_sales_group.sort_values(['HASHED_INVOICE_ID','TRANDATE'])

# COMMAND ----------

df_sales = pd.merge(df_sales_group, df_items, on='PRODUCT_ID', how='left')
df_sales = df_sales.rename(columns={'PRICE_x':'SALES_PRICE','PRICE_y':'LIST_PRICE'})
df_sales['TRAN_YEAR'] = df_sales['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[0])
df_sales['TRAN_MONTH'] = df_sales['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[1])
df_sales['TRAN_DAY'] = df_sales['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[2])

df_sales = df_sales.drop(['TRANSLATOR','HASHED_CUSTOMER_ID','SHOP_NO'], axis=1)

df_sales

# COMMAND ----------

df_sales.info()

# COMMAND ----------

df_sales_group_isbn = df_sales.groupby(['ISBN13'])['QUANTITY','AMOUNT'].sum().reset_index()

df_sales_group_isbn.sort_values('QUANTITY',ascending=False).head(50)

# COMMAND ----------

df_sales_isbn = pd.merge(df_sales_group_isbn,df_items, on='ISBN13', how='left')

df_sales_isbn

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Google API Datasets

# COMMAND ----------

# external dataset from googleapi books
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm")

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded as (
# MAGIC select 
# MAGIC     explode(items)
# MAGIC from isbn_google_reomm
# MAGIC ),
# MAGIC volumeinfo AS (
# MAGIC select 
# MAGIC     col.volumeInfo.*
# MAGIC FROM
# MAGIC     exploded
# MAGIC )
# MAGIC select 
# MAGIC lower(publisher)
# MAGIC from 
# MAGIC volumeinfo
# MAGIC where imageLinks.thumbnail is not null and categories[0] is not null 

# COMMAND ----------

df_isbn_google_reomm_cleaned = spark.sql("""
    with exploded as (
    select 
        explode(items)
    from isbn_google_reomm
    ),
    volumeinfo AS (
    select 
        col.volumeInfo.*
    FROM
        exploded
    )
    select 
    lower(authors[0]) as authors,
    lower(publisher) as publisher,
    lower(categories[0]) as categories,
    lower(description) as description,
    imageLinks.thumbnail,
    lower(title) as title,
    infoLink,
    replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn
    from 
    volumeinfo
    where imageLinks.thumbnail is not null and categories[0] is not null 
""").toPandas()

df_isbn_google_reomm_cleaned = df_isbn_google_reomm_cleaned.drop_duplicates()
df_isbn_google_reomm_cleaned = df_isbn_google_reomm_cleaned.rename(columns={'isbn':'ISBN13'})

# COMMAND ----------

df_isbn_google_reomm_cleaned.head()

# COMMAND ----------

df_isbn = pd.merge(df_sales_isbn,df_isbn_google_reomm_cleaned,on='ISBN13',how='left')

df_isbn[(df_isbn['authors'].isnull())&(df_isbn['AUTHOR'].isnull())]