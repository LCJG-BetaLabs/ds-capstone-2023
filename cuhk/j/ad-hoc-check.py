# Databricks notebook source
import os
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 21, 9

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"


df_items = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_*.csv"), header=True)
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 

df_items.createOrReplaceTempView("df_items") # spark read
df_reomm.createOrReplaceTempView("df_reomm") # spark read

df_items_p = df_items.toPandas() # padnas 
df_reomm_p = df_reomm.toPandas() # padnas 


# COMMAND ----------

# MAGIC %sql
# MAGIC with joined AS (
# MAGIC   select 
# MAGIC     *
# MAGIC   from
# MAGIC     df_reomm r
# MAGIC   left join
# MAGIC     df_items i on trim(i.ISBN13) = trim(r.ISBN13)
# MAGIC ),
# MAGIC matched_count AS (
# MAGIC   select 
# MAGIC     count(1) matched_cnt
# MAGIC   from 
# MAGIC     joined
# MAGIC   WHERE
# MAGIC     PRD_CATEGORY is not null
# MAGIC ),
# MAGIC total_count AS (
# MAGIC   select count(1) as total_cnt from df_reomm
# MAGIC )
# MAGIC select (select matched_cnt from matched_count) / (select total_cnt from total_count) * 100 as matched_rate_in_percent

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   *
# MAGIC from
# MAGIC   df_items

# COMMAND ----------

# 9781503712423


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from df_reomm where ISBN13 = 9781912707324

# COMMAND ----------

len(df_reomm_p['ISBN13'].unique())

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from df_items where ISBN13 = 9780593225905

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from df_items

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from df_items where ISBN13 = 9780340681251

# COMMAND ----------

df_items_p[df_items_p['TITLE'] == "My Very Own Big Book 80-page: PAW Patrol - Pups on a Roll"]

# COMMAND ----------

df_items_p.tail(10)

# COMMAND ----------

df_reomm_p['ISBN13'].unique()[-10:]

# COMMAND ----------

df_items_p[df_items_p['ISBN13'] == "9781743632918"]

# COMMAND ----------

!ls -ltr

# COMMAND ----------

! cat isbn_item_matching_20230420_test.json

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from df_items where ISBN13 =  9781250798954

# COMMAND ----------

df_items_p[df_items_p['PRODUCT_ID'] == 9781250798954]  

# COMMAND ----------

len(df_items_p)

# COMMAND ----------

!ls -ltr

# COMMAND ----------

! cat ./isbn_google_20230420_2.json | wc -l

# COMMAND ----------



# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_item_matching_20230420_2.json").createOrReplaceTempView("isbn_item_matching")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) / 3364 from isbn_item_matching

# COMMAND ----------

# MAGIC %sql
# MAGIC select book.* from isbn_item_matching

# COMMAND ----------

!ls -ltr

# COMMAND ----------

! date && ls -ltr

# COMMAND ----------

cat isbn_google_reomm_20230421.json | tail -n 100

# COMMAND ----------



# COMMAND ----------

!ls -ltr

# COMMAND ----------

83497172 / 1000 / 1024

# COMMAND ----------

from shutil import copyfile
output_path="isbn_google_reomm_20230421_2.json" 
output_path_e = output_path.replace(".json", "_in_progress.json")
copyfile(f'./{output_path}', f'/dbfs/{output_path_e}')
dbutils.fs.cp (f"{output_path}", "dbfs:/dbfs/")

# COMMAND ----------

dbutils.fs.ls ("/dbfs/")

# COMMAND ----------

!ls  /dbfs/ -ltr

# COMMAND ----------



# COMMAND ----------

dbutils.fs.cp (f"{output_path_e}", "dbfs:/dbfs/")

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from isbn_google_reomm

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded as (
# MAGIC   select 
# MAGIC     explode(items)
# MAGIC   from isbn_google_reomm
# MAGIC )
# MAGIC select 
# MAGIC   count(distinct col.volumeInfo.industryIdentifiers[1].identifier)
# MAGIC FROM
# MAGIC   exploded

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded as (
# MAGIC   select 
# MAGIC     explode(items)
# MAGIC   from isbn_google_reomm
# MAGIC )
# MAGIC select 
# MAGIC   count(distinct col.volumeInfo.categories)
# MAGIC FROM
# MAGIC   exploded

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded as (
# MAGIC   select 
# MAGIC     explode(items)
# MAGIC   from isbn_google_reomm
# MAGIC ),
# MAGIC volumeinfo AS (
# MAGIC   select 
# MAGIC     -- col.volumeInfo,
# MAGIC     -- col.volumeInfo.authors,
# MAGIC     -- col.volumeInfo.description as description,
# MAGIC     -- col.volumeInfo.categories[0] as categories,
# MAGIC     col.volumeInfo.*
# MAGIC   FROM
# MAGIC     exploded
# MAGIC )
# MAGIC select 
# MAGIC   authors[0] as authors,
# MAGIC   categories[0] as categories,
# MAGIC   description,
# MAGIC   imageLinks.thumbnail,
# MAGIC   industryIdentifiers[1].identifier as isbn,
# MAGIC   title
# MAGIC from 
# MAGIC   volumeinfo

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded as (
# MAGIC   select 
# MAGIC     explode(items)
# MAGIC   from isbn_google_reomm
# MAGIC ),
# MAGIC volumeinfo AS (
# MAGIC   select 
# MAGIC     -- col.volumeInfo,
# MAGIC     -- col.volumeInfo.authors,
# MAGIC     -- col.volumeInfo.description as description,
# MAGIC     -- col.volumeInfo.categories[0] as categories,
# MAGIC     col.volumeInfo.*
# MAGIC   FROM
# MAGIC     exploded
# MAGIC )
# MAGIC select 
# MAGIC   authors[0] as authors,
# MAGIC   categories[0] as categories,
# MAGIC   description,
# MAGIC   imageLinks.thumbnail,
# MAGIC   -- industryIdentifiers[1].identifier as isbn,
# MAGIC   title,
# MAGIC   infoLink,
# MAGIC   replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn
# MAGIC from 
# MAGIC   volumeinfo

# COMMAND ----------

spark.sql("""
    with exploded as (
    select 
        explode(items)
    from isbn_google_reomm
    ),
    volumeinfo AS (
    select 
        -- col.volumeInfo,
        -- col.volumeInfo.authors,
        -- col.volumeInfo.description as description,
        -- col.volumeInfo.categories[0] as categories,
        col.volumeInfo.*
    FROM
        exploded
    )
    select 
    authors[0] as authors,
    categories[0] as categories,
    description,
    imageLinks.thumbnail,
    -- industryIdentifiers[1].identifier as isbn,
    title,
    infoLink,
    replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn
    from 
    volumeinfo
""").createOrReplaceTempView("google_recomm")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from google_recomm where categories is not null 

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from google_recomm where thumbnail is not null 

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from google_recomm where thumbnail is not null and categories is not null 

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct *) from google_recomm where thumbnail is not null and categories is not null 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from google_recomm where thumbnail is not null and categories is not null 

# COMMAND ----------

! cat isbn_google_reomm_20230421_4.json | tail -n 10000

# COMMAND ----------

!ls /dbfs/team_j_downloaded_images/raw_20230423 | wc -l

# COMMAND ----------

!ls team_j_downloaded_images/raw_20230423 | wc -l

# COMMAND ----------

# image_path = "9781680522075.jpg"
# image_path = "/dbfs/team_j_downloaded_images/raw_20230423/9780007261567.jpg"

image_path = "team_j_downloaded_images/raw_20230423/9780007261567.jpg"


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load an image from a file path
img = mpimg.imread(image_path)

# Display the image using pyplot
plt.imshow(img)

# Show the plot
plt.show()

# COMMAND ----------

!ls /dbfs/

# COMMAND ----------

!ls /dbfs/team_j/image_dataset/ | wc -l

# COMMAND ----------

!ls /dbfs/team_j/image_dataset/ | wc -l

# COMMAND ----------

!ls /dbfs/team_j/image_dataset/ | wc -l

# COMMAND ----------

!ls /dbfs/team_j/image_dataset/ | wc -l

# COMMAND ----------

! ls /dbfs/team_j_downloaded_images/image_Class/

# COMMAND ----------

! ls /dbfs/team_j_downloaded_images/image_Class/LowSales/ | wc -l

# COMMAND ----------

! ls /dbfs/team_j_downloaded_images/image_Class/MediumSales/ | wc -l


# COMMAND ----------

! ls /dbfs/team_j_downloaded_images/image_Class/TopSales/ | wc -l

# COMMAND ----------

18079 + 10934 + 2665

# COMMAND ----------

!ls /dbfs/team_j/model_checkpoint_roger_binary_class_20230426/

# COMMAND ----------

from tensorflow.keras.applications import VGG16

# COMMAND ----------

!ls /dbfs/team_j/

# COMMAND ----------

!ls /dbfs/team_j/text_image_combined_model_roger_20230427_vgg16_cnn_for_text_epoch15.index/