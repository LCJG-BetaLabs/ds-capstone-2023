# Databricks notebook source
pip install BeautifulSoup4

# COMMAND ----------

pip install lxml

# COMMAND ----------

import os
import pandas as pd
import bs4
import requests
from bs4 import BeautifulSoup as bf
import json
from PIL import Image
import requests
from io import BytesIO
from IPython.display import Image 
from IPython.core.display import HTML 

# COMMAND ----------

container ="data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv"), header=True)
#df = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv"), header=True)
df = df.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

#https://www.amazon.com/First-100-Pretty-Pink-Words/dp/1783931833/ref=sr_1_1?Adv-Srch-Books-Submit.x=37&Adv-Srch-Books-Submit.y=9&qid=1680447679&refinements=p_66%3A9781783931835&s=books&sr=1-1&unfiltered=1

# COMMAND ----------

HEADERS = ({'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})

# COMMAND ----------

headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0'}

# COMMAND ----------

# URL = "https://www.amazon.com/Cool-Gross-Jokes-Kit/dp/1488902844/ref=sr_1_1?Adv-Srch-Books-Submit.x=37&Adv-Srch-Books-Submit.y=9&qid=1680447746&refinements=p_66%3A9781488902840&s=books&sr=1-1&unfiltered=1"
URL = "https://isbnsearch.org/isbn/9781783931835"
webpage = requests.get(URL, headers=headers)

# COMMAND ----------

webpage.content

# COMMAND ----------

soup = bf(webpage.content, "lxml")
# soup = bf(webpage.content, "html.parser")

# COMMAND ----------

soup

# COMMAND ----------

pic = soup.find("div", attrs={'class':'image'})
pic

# COMMAND ----------

pic.img.get('src')

# COMMAND ----------

imgs_str = pic.img.get('src')
# imgs_dict = json.loads(imgs_str)
# num_element = 0 
# first_link = list(imgs_dict.keys())[num_element]

# COMMAND ----------

Image(url= imgs_str)

# COMMAND ----------

response = requests.get(first_link)
img = Image.open(BytesIO(response.content))

# COMMAND ----------

productdetail = soup.find_all("div", attrs={'id':'detailBulletsWrapper_feature_div'})
productdetail