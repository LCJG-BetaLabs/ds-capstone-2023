# Databricks notebook source
# MAGIC %md # Package & dataframe (spark + pandas) initiated 

# COMMAND ----------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
import seaborn as sns
from datetime import datetime   
from shutil import copyfile
import yake

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

# MAGIC %md ## df preview & check before running data fetcher function
# MAGIC   - get the distinct ISBN code from `*RECOMMENDATION_*` csv file 
# MAGIC   - and then get the external data points from ISBNDB and gogolebooksapi 

# COMMAND ----------

df_items_p

# COMMAND ----------

# MAGIC %md # External Source 1 - ISBNDB

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

# MAGIC %md # Function called to send request and store to azure persistent storage

# COMMAND ----------

call_api_and_write_to_file(
    isbn_list=df_items_p['ISBN13'][:].to_list(), 
    output_path="isbn_item_matching_20230420_2.json"
)

# COMMAND ----------

# MAGIC %md # list and quick peek dbfs

# COMMAND ----------

dbutils.fs.ls ("/dbfs/")

# COMMAND ----------

# MAGIC %md # copy the result (saved in json format) to `/dbfs`

# COMMAND ----------

dbutils.fs.cp ("isbn_item_matching_20230420_2.json", "dbfs:/dbfs/")

# COMMAND ----------

# MAGIC %md # list the `/dbfs` again to make sure the file is in place

# COMMAND ----------

dbutils.fs.ls("/dbfs/")

# COMMAND ----------

# MAGIC %md # create a spark temp view to make the .json file querable with SparkSQL

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_item_matching_20230420_2.json").createOrReplaceTempView("isbn_item_matching")

# COMMAND ----------

# MAGIC %md # quick peek the target field 

# COMMAND ----------

# MAGIC %sql
# MAGIC select book.image from isbn_item_matching

# COMMAND ----------

# MAGIC %md # count the non-null book.image rows 

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from isbn_item_matching where book.image is not null

# COMMAND ----------

# MAGIC %md # quick peek at the nested field `book`

# COMMAND ----------

# MAGIC %sql
# MAGIC select book.* from isbn_item_matching

# COMMAND ----------

import os
os.listdir('/dbfs/')


# COMMAND ----------

from shutil import copyfile

# COMMAND ----------

# https://academia.stackexchange.com/questions/164349/is-it-possible-to-get-a-book-category-based-on-isbn

# COMMAND ----------

# MAGIC %md # External Source 2 - googleapis
# MAGIC - 

# COMMAND ----------

import requests, pprint, json, time

# COMMAND ----------

# MAGIC %md # function that fetches googleads api result and store in azure persistent storage

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

# MAGIC %md #  call the function

# COMMAND ----------

call_api_and_write_to_file_append(
    isbn_list=df_items_p['ISBN13'][:].to_list()[:],
    output_path="isbn_google_20230420_2.json"
)

# COMMAND ----------

# MAGIC %md # quick peek at the saved json file

# COMMAND ----------

!cat isbn_google_test.json

# COMMAND ----------

dbutils.fs.cp ("isbn_google_20230420_2.json", "dbfs:/dbfs/")
# dbutils.fs.cp ("", "dbfs:/dbfs/")

# COMMAND ----------

# MAGIC %md # copy the file from driver node to azure storage

# COMMAND ----------

copyfile('./isbn_google_20230420_2.json', '/dbfs/isbn_google_20230420_2.json')

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_20230420_2.json").createOrReplaceTempView("isbn_google")

# COMMAND ----------

# MAGIC %md # make the fetehed json file querable on spark and have a look at it

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

# MAGIC %md # expand nested rows to check 

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

# MAGIC %md # check row count and the proportiion on non-null categories field 

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

# MAGIC %md # the below code is the 2nd batch run on the fetching result from googlebooksapis
# MAGIC   - because the first run stopped and needs to be resumed.

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

