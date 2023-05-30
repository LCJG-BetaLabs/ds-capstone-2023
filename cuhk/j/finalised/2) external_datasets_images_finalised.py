# Databricks notebook source
# MAGIC %md # Two seperate runs that fetches books details from google api
# MAGIC - after checking both the external dataset from ISBNDB and google books api, we find that the result from the latter one is more comprehensive,
# MAGIC - and thus we decide to fectch the image data for later-on model training

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm_20230421_2")
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_4.json").createOrReplaceTempView("isbn_google_reomm_20230421_4")

# COMMAND ----------

# MAGIC %md # get isbn and image link pair (excluding rows with null image link )

# COMMAND ----------

# MAGIC %md #### 32071 image links found from google api

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded_2 as (
# MAGIC   select 
# MAGIC       explode(items)
# MAGIC   from isbn_google_reomm_20230421_2
# MAGIC ),
# MAGIC volumeinfo_2 AS (
# MAGIC   select 
# MAGIC   col.volumeInfo.*
# MAGIC   FROM
# MAGIC       exploded_2
# MAGIC ),
# MAGIC exploded_4 as (
# MAGIC   select 
# MAGIC       explode(items)
# MAGIC   from isbn_google_reomm_20230421_4
# MAGIC ),
# MAGIC volumeinfo_4 AS (
# MAGIC   select 
# MAGIC   col.volumeInfo.*
# MAGIC   FROM
# MAGIC       exploded_2
# MAGIC ),
# MAGIC unioned AS (
# MAGIC   select * from volumeinfo_2
# MAGIC   union all 
# MAGIC   select * from volumeinfo_4
# MAGIC ),
# MAGIC isbn_image_link_pair AS (
# MAGIC   SELECT
# MAGIC     distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
# MAGIC     imageLinks.thumbnail as thumbnail
# MAGIC   FROM
# MAGIC     unioned
# MAGIC )
# MAGIC select 
# MAGIC   count(1)
# MAGIC from 
# MAGIC   isbn_image_link_pair
# MAGIC where 
# MAGIC   thumbnail is not null 

# COMMAND ----------

# MAGIC %md # create a list that stores the ISBN and thumbnail link pair

# COMMAND ----------

isbn_thumbnail_pairs = spark.sql("""
    with exploded_2 as (
    select 
        explode(items)
    from isbn_google_reomm_20230421_2
    ),
    volumeinfo_2 AS (
    select 
    col.volumeInfo.*
    FROM
        exploded_2
    ),
    exploded_4 as (
    select 
        explode(items)
    from isbn_google_reomm_20230421_4
    ),
    volumeinfo_4 AS (
    select 
    col.volumeInfo.*
    FROM
        exploded_2
    ),
    unioned AS (
    select * from volumeinfo_2
    union all 
    select * from volumeinfo_4
    ),
    isbn_image_link_pair AS (
    SELECT
        distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
        imageLinks.thumbnail as thumbnail
    FROM
        unioned
    )
    select 
        *
    from 
    isbn_image_link_pair
    where 
    thumbnail is not null 
""").rdd.map(lambda x:{x['isbn']:x['thumbnail']}).collect()


# COMMAND ----------

# MAGIC %md # quick check on the list isbn_thumbnail_pairs

# COMMAND ----------

for item in isbn_thumbnail_pairs[:5]:
    print(list(item.keys())[0])

# COMMAND ----------

!/databricks/python3/bin/python -m pip install --upgrade pip

# COMMAND ----------

!pip install asyncio aiohttp

# COMMAND ----------

from shutil import copyfile

# COMMAND ----------

# MAGIC %md # async function that fetch the book cover image and stores in azure persistent storage 

# COMMAND ----------

import asyncio
import aiohttp
import os


target_dir = "./team_j_downloaded_images/raw_20230423/"

# Check if the directory exists
if os.path.exists(target_dir):
    print('Directory exists')
else:
    # Create the directory if it doesn't exist
    os.makedirs(target_dir)
    print(f'Directory {target_dir} created')


async def download_image(url, filename, retries=3):
    async with aiohttp.ClientSession() as session:
        while retries > 0:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(filename, 'wb') as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    return
                else:
                    print(f'Response status was {response.status}, {retries} retries left, retrying...')
                    retries -= 1
                    await asyncio.sleep(1)
        print(f'Failed to download {url} after {retries} retries.')

async def run():

    #url = 'http://books.google.com/books/content?id=vQKPswEACAAJ&printsec=frontcover&img=1&zoom=1&source=gbs_api'
    #filename = f'{target_dir}9781680522075.jpg'
    #await download_image(url, filename)
    #print(f'Saved {filename} successfully!')

    for item in isbn_thumbnail_pairs[:]:
        for isbn, url in item.items():
            #print(isbn, url)
            filename = f'{target_dir}{isbn}.jpg'
            await download_image(url, filename)
            print(f'Saved {filename} successfully!')

asyncio.run(run())



# COMMAND ----------

# MAGIC %md # move the fetched image files from driver node to azure storage

# COMMAND ----------


src_dir = target_dir
dest_dir = f'/dbfs/{src_dir.split("./")[1]}'

# Check if the directory exists
if os.path.exists(dest_dir):
    print(f'Directory {dest_dir} exists')
else:
    # Create the directory if it doesn't exist
    os.makedirs(dest_dir)
    print(f'Directory {dest_dir} created')


file_list = os.listdir(src_dir)

for file_name in file_list:
    # Construct the full file paths
    src_file = os.path.join(src_dir, file_name)
    dest_file = os.path.join(dest_dir, file_name)

    # Copy the file using shutil.copyfile()
    copyfile(src_file, dest_file)
    print(f"copied file {file_name} to file path : {dest_file}")

# COMMAND ----------

# MAGIC %md # inspect fetched images

# COMMAND ----------

!ls team_j_downloaded_images/raw_20230423 | wc -l

# COMMAND ----------

!ls /dbfs/team_j_downloaded_images/raw_20230423 | wc -l

# COMMAND ----------

# MAGIC %md #### only successfully 31747 images from the thumbnail links

# COMMAND ----------

# MAGIC %md ##### show a few images with isbn

# COMMAND ----------

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from IPython.core.display import HTML

from IPython.display import display

image_dir = "/dbfs/team_j_downloaded_images/raw_20230423/"
raw_image_list = os.listdir(f'{image_dir}')


file_paths = raw_image_list[:30]

# Calculate the number of rows and columns needed to display the images in a grid
num_images = len(file_paths)
num_cols = 5
num_rows = math.ceil(num_images / num_cols)

# Create a new figure with the appropriate size
fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, 20), gridspec_kw={'hspace': 0.4, 'wspace': 0})
# ax = ax.ravel()

# Loop through the file paths and display each image in a subplot
for i, file_path in enumerate(file_paths):
    row = i // num_cols
    col = i % num_cols
    img = mpimg.imread(f"{image_dir}{file_path}")
    title = file_path
    ax[row][col].imshow(img)
    ax[row][col].set_title(title)
    ax[row][col].axis('off')

# Hide any unused subplots
for i in range(num_images, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    ax[row][col].axis('off')

# Display the plot
html = HTML(f'<div style="overflow-x: scroll; width: 100%;">{fig}</div>')
plt.show()
