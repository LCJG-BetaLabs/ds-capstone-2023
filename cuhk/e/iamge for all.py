# Databricks notebook source
import os
import pandas as pd
import requests
import json
from PIL import Image
import requests
from io import BytesIO
import time
import urllib.request

# COMMAND ----------

container ="data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv"), header=True)
df2 = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv"), header=True)
df = df.toPandas()
df2 = df2.toPandas()

# COMMAND ----------

df

# COMMAND ----------

HEADERS = ({'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})

# COMMAND ----------

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
ua=UserAgent()
hdr = {'User-Agent': ua.random,
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
      'Accept-Encoding': 'none',
      'Accept-Language': 'en-US,en;q=0.8',
      'Connection': 'keep-alive'}

# COMMAND ----------

ISBN13 = df['ISBN13']
len(ISBN13)

# COMMAND ----------

ISBN13[0:20]

# COMMAND ----------

# urllib.request.urlretrieve("https://covers.openlibrary.org/b/isbn/9781783931835-S.jpg", "/dbfs/FileStore/9781783931835.jpg")
url = "https://covers.openlibrary.org/b/isbn/$ISBN-S.jpg"
path = "/dbfs/FileStore/$ISBN.jpg"
print(url.replace("$ISBN","9781783931835" ))
print(path.replace("$ISBN","9781783931835" ))


# COMMAND ----------

from operator import is_not
from functools import partial
ISBN13 = ISBN13.dropna()
ISBN13 = list(filter(partial(is_not, None), ISBN13))


# COMMAND ----------

ISBN13[0:50]

# COMMAND ----------

url = "https://covers.openlibrary.org/b/isbn/$ISBN-M.jpg"
path = "/dbfs/FileStore/Bookimages2022/$ISBN.jpg"
i = 0
for isbn in ISBN13[205:-1]:
    print(url.replace("$ISBN",isbn.strip()))
    print(path.replace("$ISBN",isbn.strip()))
    urllib.request.urlretrieve(url.replace("$ISBN",isbn.strip() ), path.replace("$ISBN",isbn.strip() ))
    print(i)
    i = i+1
    time.sleep(10)

# COMMAND ----------

ISBN13[205:-1]

# COMMAND ----------

display(Image(url= "https://blog.finxter.com/wp-content/uploads/2022/04/greenland_02a.jpg"))

# COMMAND ----------

display(dbutils.fs.ls("/FileStore/Bookimages"))
# display(dbutils.fs.ls("/FileStore/Bookimages2022"))

# COMMAND ----------

dbutils.fs.mkdirs("/FileStore/Bookimages2022")

# COMMAND ----------

displayHTML("<img src ='/files/Bookimages2022/9780375705922.jpg'>")

# COMMAND ----------

import matplotlib.pyplot as plt
plt.scatter(x=[1,2,3], y=[2,4,3])
plt.savefig('/dbfs/FileStore/figure.png')

# COMMAND ----------

urllib.request.urlretrieve("https://covers.openlibrary.org/b/isbn/9781783931835-S.jpg", "/dbfs/FileStore/bookImages/11119781783931835.jpg")