# Databricks notebook source
import os
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
import seaborn as sns
from shutil import copyfile

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 21, 9

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"


df_items = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_*.csv"), header=True)
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 

# df_items.createOrReplaceTempView("df_items") # spark read
# df_reomm.createOrReplaceTempView("df_reomm") # spark read

df_items_p = df_items.toPandas() # padnas 
df_reomm_p = df_reomm.toPandas() # padnas 


# COMMAND ----------

df_items_p

# COMMAND ----------

# import requests as req

# ISBNDB_KEY = "49651_00d73437d983d0680e84d48aa1c39a3d"
# h = {'Authorization': ISBNDB_KEY}
# resp = req.get(f"https://api2.isbndb.com/book/9781743632918", headers=h)
# print(resp.json())

# COMMAND ----------

# import requests
# import json

# ISBNDB_KEY = "49651_00d73437d983d0680e84d48aa1c39a3d"
# h = {'Authorization': ISBNDB_KEY}

# def call_api_and_write_to_file(isbn_list, output_path):
#     # Create an empty list to hold the response data from each API URL
#     data_list = []
    
#     # Loop through each API URL in the list
#     for isbn_i in isbn_list:
#         # Make a GET request to the API endpoint
#         #response = requests.get(url)
#         url = f"https://api2.isbndb.com/book/{isbn_i}"
#         response = requests.get(url, headers=h)

#         # Check if the request was successful
#         if response.status_code == 200:
#             # Parse the JSON response data
#             data = json.loads(response.text)
            
#             # Append the JSON data to the list
#             data_list.append(data)
            
#             print(f"Data from {url} written to data.json")
#         else:
#             print(f"Request to {url} failed with status code: {response.status_code}")

#     # Open a file for writing
#     with open(output_path, 'w') as f:
#         # Write the JSON data from all API URLs to the file
#         json.dump(data_list, f)
        
#     print(f"All data written to {output_path}")

# COMMAND ----------

# MAGIC %md # source: isbndb

# COMMAND ----------

import time
import requests
import json

def call_api_and_write_to_file(isbn_list, output_path, max_retries=3):
    ISBNDB_KEY = "49651_00d73437d983d0680e84d48aa1c39a3d"
    h = {'Authorization': ISBNDB_KEY}
    
    # Create an empty list to hold the response data from each API URL
    data_list = []

    # Set the maximum number of retries
    #max_retries = 3

    # Loop through each API URL in the list
    for isbn_i in isbn_list:
        # Initialize the retry count
        retries = 0

        # Make a GET request to the API endpoint
        url = f"https://api2.isbndb.com/book/{isbn_i}"

        response = requests.get(url, headers=h)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response data
            data = json.loads(response.text)
            
            # Append the JSON data to the list
            data_list.append(data)
            
            print(f"Data from {url} written to {output_path}")
        elif response.status_code == 403:
            while retries < max_retries:
                retries += 1
                print(f"Request to {url} failed with status code: {response.status_code}. Retrying in 10 seconds...")
                time.sleep(10)
                response = requests.get(url, headers=h)
                if response.status_code == 200:
                    data = json.loads(response.text)
                    data_list.append(data)
                    print(f"Data from {url} written to{output_path}")
                    break
                elif retries == max_retries:
                    print(f"Maximum number of retries reached. Retry failed with status code: {response.status_code}")
                else:
                    print(f"Retry failed with status code: {response.status_code}. Retrying...")
        else:
            print(f"Request to {url} failed with status code: {response.status_code}")

    # Open a file for writing
    with open(output_path, 'w') as f:
        # Write the JSON data from all API URLs to the file
        json.dump(data_list, f)
        
    print(f"All data written to {output_path}")


# COMMAND ----------

call_api_and_write_to_file(
    isbn_list=df_items_p['ISBN13'][:].to_list(), 
    output_path="isbn_item_matching_20230420_2.json"
)

# COMMAND ----------

!ls -ltr

# COMMAND ----------

dbutils.fs.ls ("/dbfs/")

# COMMAND ----------

dbutils.fs.cp ("isbn_item_matching_20230420_2.json", "dbfs:/dbfs/")

# COMMAND ----------

dbutils.fs.ls("/dbfs/")

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_item_matching_20230420_2.json").createOrReplaceTempView("isbn_item_matching")

# COMMAND ----------

# MAGIC %sql
# MAGIC select book.image from isbn_item_matching

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from isbn_item_matching where book.image is not null

# COMMAND ----------

# MAGIC %sql
# MAGIC select book.* from isbn_item_matching

# COMMAND ----------

# 

# COMMAND ----------

import os
os.listdir('/dbfs/')


# COMMAND ----------

from shutil import copyfile

# COMMAND ----------

# copyfile('./isbn_item_matching_20230420_2.json', '/dbfs/isbn_item_matching_20230420_2.json')

# COMMAND ----------

# len(df_reomm_p['ISBN13'].unique())

# COMMAND ----------

# https://academia.stackexchange.com/questions/164349/is-it-possible-to-get-a-book-category-based-on-isbn

# COMMAND ----------

# MAGIC %md # source: googleapis

# COMMAND ----------

import requests, pprint, json, time

# COMMAND ----------


isbn = "9781474995795"
url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
resp = requests.get(url)
pprint.pprint(resp.json())

# COMMAND ----------

def call_api_and_write_to_file_append(isbn_list, output_path, max_retries=6):
    # Set the maximum number of retries
    #max_retries = 3

    # Loop through each API URL in the list
    for isbn_i in isbn_list:
        # Initialize the retry count
        retries = 0

        # Make a GET request to the API endpoint
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn_i}"
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response data
            data = json.loads(response.text)
            
            # Open the output file in append mode and write the JSON data
            with open(output_path, 'a') as f:
                json.dump(data, f)
                f.write('\n')  # Add a newline character to separate each JSON object
            
            print(f"Data from {url} written to {output_path}")
        elif response.status_code == 403 or response.status_code == 429:
            while retries < max_retries:
                retries += 1
                print(f"Request to {url} failed with status code: {response.status_code}. Retrying in 10 seconds...")
                time.sleep(10)
                response = requests.get(url)
                if response.status_code == 200:
                    data = json.loads(response.text)
                    with open(output_path, 'a') as f:
                        json.dump(data, f)
                        f.write('\n')
                    print(f"Data from {url} written to{output_path}")
                    break
                elif retries == max_retries:
                    print(f"Maximum number of retries reached. Retry failed with status code: {response.status_code}")
                else:
                    print(f"Retry failed with status code: {response.status_code}. Retrying...")
        else:
            print(f"Request to {url} failed with status code: {response.status_code}")

    print(f"All data written to {output_path}")


# COMMAND ----------

# 

# COMMAND ----------

call_api_and_write_to_file_append(
    isbn_list=df_items_p['ISBN13'][:].to_list()[:],
    output_path="isbn_google_20230420_2.json"
)

# COMMAND ----------

!cat isbn_google_test.json

# COMMAND ----------

!ls

# COMMAND ----------

dbutils.fs.cp ("isbn_google_20230420_2.json", "dbfs:/dbfs/")
# dbutils.fs.cp ("", "dbfs:/dbfs/")

# COMMAND ----------

!ls

# COMMAND ----------

# copyfile('./isbn_item_matching_20230420_2.json', '/dbfs/isbn_item_matching_20230420_2.json')

# COMMAND ----------

copyfile('./isbn_google_20230420_2.json', '/dbfs/isbn_google_20230420_2.json')

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_20230420_2.json").createOrReplaceTempView("isbn_google")

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded as (
# MAGIC   select 
# MAGIC     explode(items)
# MAGIC   from 
# MAGIC     isbn_google
# MAGIC )
# MAGIC select
# MAGIC   col.*
# MAGIC from 
# MAGIC   exploded

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded as (
# MAGIC   select 
# MAGIC     explode(items)
# MAGIC   from 
# MAGIC     isbn_google
# MAGIC )
# MAGIC select
# MAGIC   col.volumeInfo.categories, 
# MAGIC   col.volumeInfo.imageLinks.smallThumbnail,
# MAGIC   col.volumeInfo.imageLinks.thumbnail
# MAGIC
# MAGIC from 
# MAGIC   exploded

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded as (
# MAGIC   select 
# MAGIC     explode(items)
# MAGIC   from 
# MAGIC     isbn_google
# MAGIC )
# MAGIC select
# MAGIC   --col.volumeInfo.categories, 
# MAGIC   --col.volumeInfo.imageLinks.smallThumbnail,
# MAGIC   --col.volumeInfo.imageLinks.thumbnail
# MAGIC
# MAGIC   -- col.volumeInfo.categories
# MAGIC   count(1) / 3364
# MAGIC from 
# MAGIC   exploded
# MAGIC where col.volumeInfo.categories is not null

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

output_path="isbn_google_reomm_20230421_2.json" 

call_api_and_write_to_file_append(
    isbn_list=df_reomm_p['ISBN13'].unique().tolist()[:],
    output_path=output_path
    # output_path="/dbfs/isbn_google_reomm_20230421_2.json"
)

copyfile(f'./{output_path}', f'/dbfs/{output_path}')
dbutils.fs.cp (f"{output_path}", "dbfs:/dbfs/")

# COMMAND ----------

copyfile(f'./{output_path}', f'/dbfs/{output_path}')
dbutils.fs.cp (f"{output_path}", "dbfs:/dbfs/")

# COMMAND ----------


# copyfile('./isbn_item_matching_20230420_2.json', '/dbfs/isbn_item_matching_20230420_2.json')

# COMMAND ----------

# dbutils.fs.cp ("isbn_item_matching_20230420_2.json", "dbfs:/dbfs/")

# COMMAND ----------

# Data from https://www.googleapis.com/books/v1/volumes?q=isbn:9780603566769  written to isbn_google_reomm_20230421.json

# COMMAND ----------

!ls /dbfs -ltr

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm")

# COMMAND ----------

remaining_isbn_to_fetech_from_google = spark.sql("""
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
    ),
    already_fetched_isbn_google as (
    select 
        distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn
    from 
    volumeinfo
    )
    select 
    trim(ISBN13) as not_found_isbn
    --count(1)
    FROM
    df_reomm
    where
    trim(ISBN13) not in (select isbn from already_fetched_isbn_google)

""").rdd.map(lambda x:x["not_found_isbn"]).collect()

# COMMAND ----------

len(remaining_isbn_to_fetech_from_google)

# COMMAND ----------

output_path="isbn_google_reomm_20230421_4.json" 

call_api_and_write_to_file_append(
    isbn_list=remaining_isbn_to_fetech_from_google,
    output_path=output_path
    # output_path="/dbfs/isbn_google_reomm_20230421_2.json"
)

copyfile(f'./{output_path}', f'/dbfs/{output_path}')
dbutils.fs.cp (f"{output_path}", "dbfs:/dbfs/")

# COMMAND ----------

!ls /dbfs -ltr

# COMMAND ----------

