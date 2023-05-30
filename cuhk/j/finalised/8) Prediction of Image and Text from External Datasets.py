# Databricks notebook source
# MAGIC %md # load the registered model on mlflow

# COMMAND ----------

import mlflow.keras
import numpy as np
import pandas as pd
import os

# # Load the registered model
# model_uri = 'runs:/5b5cda147afb42388d4aaeea222b46c1/model'
# loaded_model = mlflow.keras.load_model(model_uri)


# COMMAND ----------

# MAGIC %md # load the tokenizer for text embedding  

# COMMAND ----------

!pip install transformers

# COMMAND ----------

from transformers import BertTokenizer
import numpy as np

# COMMAND ----------

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# COMMAND ----------

text = "This is a sample text"
tokens = tokenizer.encode(text, add_special_tokens=True)

# COMMAND ----------

MAX_LEN = 512
padded_tokens = tokens + [0] * (MAX_LEN - len(tokens))

# COMMAND ----------

text_data = np.array([padded_tokens])

# COMMAND ----------

text_data

# COMMAND ----------

image_data = np.random.rand(1, 128, 128, 3)

# COMMAND ----------

image_data.shape

# COMMAND ----------

# loaded_model

# Make predictions using the loaded model
predictions = loaded_model.predict([image_data, text_data])


# COMMAND ----------

predictions

# COMMAND ----------

# MAGIC %md # predict in batch

# COMMAND ----------

!ls /dbfs/team_j/image_dataset

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm_20230421_2")
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_4.json").createOrReplaceTempView("isbn_google_reomm_20230421_4")

# COMMAND ----------



# COMMAND ----------

isbn_desc = spark.sql("""
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
        imageLinks.thumbnail as thumbnail,
        description
    FROM
        unioned
    )
    select 
    isbn,
    description
    from 
    isbn_image_link_pair
    where 
    thumbnail is not null and description is not null 
""").toPandas()

# COMMAND ----------

isbn_desc

# COMMAND ----------

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 
df_reomm_p = df_reomm.toPandas() # padnas 


# COMMAND ----------

def clean_recomm_df(df: pd.DataFrame) -> pd.DataFrame:

    # df_reomm_p_2 = df_reomm_p[df_reomm_p["HASHED_INVOICE_ID"].apply(lambda s: s.startswith("0x"))]
    # df_reomm_p_2 = df_reomm_p_2[~df_reomm_p_2['QUANTITY'].isnull()]
    # df_reomm_p_2 = df_reomm_p_2.drop("ISBN13", axis=1)

    # df_reomm_p_2["PRICE"] = df_reomm_p_2["PRICE"].astype(float)
    # df_reomm_p_2["QUANTITY"] = df_reomm_p_2["QUANTITY"].astype(int)
    # df_reomm_p_2["AMOUNT"] = df_reomm_p_2["AMOUNT"].astype(float)

    # df_reomm_p_2["year"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[0])
    # df_reomm_p_2["month"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[1])
    # df_reomm_p_2["day"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[2])


    ########################################################################################################################################################
    #   ref: https://adb-5911062106551859.19.azuredatabricks.net/?o=5911062106551859#notebook/3108408038812593/command/751034215087416                     #
    ########################################################################################################################################################

    df_2 = df[df["HASHED_INVOICE_ID"].apply(lambda s: s.startswith("0x"))]
    df_2 = df_2[~df_2['QUANTITY'].isnull()]
    #df_2 = df_2.drop("ISBN13", axis=1)

    df_2['ISBN13'] = df_2['ISBN13'].apply(lambda s:s.rstrip())

    df_2["PRICE"] = df_2["PRICE"].astype(float)
    df_2["QUANTITY"] = df_2["QUANTITY"].astype(int)
    df_2["AMOUNT"] = df_2["AMOUNT"].astype(float)

    df_2["year"] = df_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[0])
    df_2["month"] = df_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[1])
    df_2["day"] = df_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[2])

    return df_2

# COMMAND ----------

df_reomm_p_cleaned = clean_recomm_df(df = df_reomm_p)
df_reomm_p_cleaned_wo_coupon = df_reomm_p_cleaned[df_reomm_p_cleaned['TITLE'] != "Group Cash Coupon - $100"]  # exclude this item
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon.groupby("ISBN13").sum().reset_index()
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon_stat_by_isbn.sort_values(by='QUANTITY', ascending=False)

q_65 = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].quantile(0.65)
df_reomm_p_cleaned_wo_coupon_stat_by_isbn['is_high_sales_volume'] = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].apply(lambda x: 1 if x >= q_65 else 0)

# COMMAND ----------

q_65 = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].quantile(0.65)
df_reomm_p_cleaned_wo_coupon_stat_by_isbn['is_high_sales_volume'] = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].apply(lambda x: 1 if x >= q_65 else 0)

# COMMAND ----------

df_text_image_dataset = pd.merge(isbn_desc, df_reomm_p_cleaned_wo_coupon_stat_by_isbn, left_on='isbn', right_on='ISBN13', how='inner')

# COMMAND ----------

df_text_image_dataset['isbn'] = df_text_image_dataset['isbn'].apply(lambda x: x + ".jpg")
df_text_image_dataset = df_text_image_dataset[['isbn', 'description', 'is_high_sales_volume']]

# COMMAND ----------



# COMMAND ----------

df_text_image_dataset[df_text_image_dataset['isbn'] == "9789812616630.jpg"]

# COMMAND ----------

text = df_text_image_dataset[df_text_image_dataset['isbn'] == "9789812616630.jpg"]['description'].values[0]

# COMMAND ----------



# COMMAND ----------

tokens = tokenizer.encode(text, add_special_tokens=True)
MAX_LEN = 512
padded_tokens = tokens + [0] * (MAX_LEN - len(tokens))
text_data = np.array([padded_tokens])

# text_data = np.array([padded_tokens])

# COMMAND ----------



# COMMAND ----------

import tensorflow as tf

# COMMAND ----------

# image_path = "/dbfs/team_j/image_dataset/9789812616630.jpg"

# img = tf.io.read_file(image_path)
# img = tf.image.decode_jpeg(img, channels=3)
# img = tf.image.resize(img, (128, 128))
# image_data = np.array(img)

image_path = "/dbfs/team_j/image_dataset/9789812616630.jpg"

img = tf.io.read_file(image_path)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, (128, 128))
image_data = np.array(img)
image_data = image_data[np.newaxis, :]

# COMMAND ----------

image_data.shape

# COMMAND ----------

predictions = loaded_model.predict([image_data, text_data])
predictions

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # download competitors's product 

# COMMAND ----------

# 9780593428979
# 9781338715422
# 9780593526774
# 9781250852564
# 9781338775891

# COMMAND ----------

!ls /dbfs/team_j/

# COMMAND ----------

import requests
import json

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

output_path="competitor_isbn_20230427_3.json"
call_api_and_write_to_file_append(
    isbn_list=[
        "9780593428979",
        "9781338715422",
        "9780593526774",
        "9781250852564",
        "9781338775891",
        "9781405294041",
        "9781338762587",
        "9781368084802",
        "9789813372283",
        "9781616559533",
        "9781338660425",
        "9781338770377"
    ],
    output_path=output_path
)

# COMMAND ----------

!ls 

# COMMAND ----------

dbutils.fs.ls ("/dbfs/")

# COMMAND ----------

from shutil import copyfile

copyfile('competitor_isbn_20230427_3.json', '/dbfs/competitor_isbn_20230427_3.json')
dbutils.fs.cp ('competitor_isbn_20230427_3.json', "dbfs:/dbfs/")

# COMMAND ----------



# COMMAND ----------

! ls /dbfs/team_j

# COMMAND ----------

spark.read.json("dbfs:/dbfs/competitor_isbn_20230427_3.json").createOrReplaceTempView("competitor_books")

# COMMAND ----------

isbn_thumbnail_pairs = spark.sql("""
  with exploded_2 as (
  select 
      explode(items)
  from competitor_books
  ),
  volumeinfo_2 AS (
  select 
  col.volumeInfo.*
  FROM
      exploded_2
  )
  select 
  distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
  imageLinks.thumbnail as thumbnail,
  description
  from volumeinfo_2
  where imageLinks.thumbnail is not null  
""").rdd.map(lambda x:{x['isbn']:x['thumbnail']}).collect()


# COMMAND ----------

competitor_product_info = spark.sql("""
  with exploded_2 as (
  select 
      explode(items)
  from competitor_books
  ),
  volumeinfo_2 AS (
  select 
  col.volumeInfo.*
  FROM
      exploded_2
  )
  select 
  distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
  imageLinks.thumbnail as thumbnail,
  description
  from volumeinfo_2
  where imageLinks.thumbnail is not null  
""").toPandas()

# COMMAND ----------

competitor_product_info

# COMMAND ----------

!ls ./team_j_downloaded_images/raw_20230427_competitor/

# COMMAND ----------

!pip install aiohttp

# COMMAND ----------

import asyncio
import aiohttp
import os


target_dir = "./team_j_downloaded_images/raw_20230427_competitor/"

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

! ls /dbfs/team_j_downloaded_images/raw_20230427_competitor/

# COMMAND ----------

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from IPython.core.display import HTML

from IPython.display import display

image_dir = "/dbfs/team_j_downloaded_images/raw_20230427_competitor/"
raw_image_list = os.listdir(f'{image_dir}')


file_paths = raw_image_list

# Calculate the number of rows and columns needed to display the images in a grid
num_images = len(file_paths)
num_cols = 2
num_rows = math.ceil(num_images / num_cols)

# Create a new figure with the appropriate size
fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, 20))
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

# Display the plot
# fig.tight_layout()
# display(fig)

# COMMAND ----------

def prepare_text_input(input_str):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.encode(text, add_special_tokens=True)
    MAX_LEN = 512
    padded_tokens = tokens + [0] * (MAX_LEN - len(tokens))
    text_data = np.array([padded_tokens])

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ProductPredictor:
    MAX_LEN = 512
    def __init__(self, model_uri='runs:/5b5cda147afb42388d4aaeea222b46c1/model'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model_uri = 'runs:/5b5cda147afb42388d4aaeea222b46c1/model'
        self.loaded_model = mlflow.keras.load_model(model_uri)

        #self.text_input = text_input
        #self.image_path = image_path
        #self.text_data = None

    def predict(self, text_input, image_path):
        tokens = self.tokenizer.encode(text_input, add_special_tokens=True)
        padded_tokens = tokens + [0] * (self.MAX_LEN - len(tokens))
        text_data = np.array([padded_tokens])


        # image_path = "/dbfs/team_j/image_dataset/9789812616630.jpg"

        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (128, 128))
        image_data = np.array(img)
        image_data = image_data[np.newaxis, :]

        predictions = self.loaded_model.predict([image_data, text_data])
        predictions_val = predictions[0][0]
        res = "high_sales" if predictions_val >=0.5 else "low_sales"

        image_name = image_path.split('/')[-1]

        #return predictions[0][0]

        # Load an image from a file path
        img = mpimg.imread(image_path)

        # Display the image using pyplot and add a title
        
        plt.imshow(img)
        plt.title(f"{image_name} | Prob: {predictions_val} | result: {res}")
        # Show the plot
        plt.show()
        print(f"description: {text_input}")

# COMMAND ----------



# COMMAND ----------

p1 = ProductPredictor(model_uri='runs:/5b5cda147afb42388d4aaeea222b46c1/model')

# COMMAND ----------

p1

# COMMAND ----------

index = 0

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 1

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 2

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 3

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 4

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 5

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 6

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 7

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 8

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 9

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------

index = 10

image_path = f"/dbfs/team_j_downloaded_images/raw_20230427_competitor/{competitor_product_info.iloc[index]['isbn']}.jpg"

text_input = competitor_product_info.iloc[index]['description']
p1.predict(
    image_path=image_path,
    text_input=text_input
)

# COMMAND ----------



# COMMAND ----------

