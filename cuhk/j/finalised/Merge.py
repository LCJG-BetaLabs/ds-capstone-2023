# Databricks notebook source
# MAGIC %md
# MAGIC # Install and import packages

# COMMAND ----------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
import warnings
import seaborn as sns
from datetime import datetime
from shutil import copyfile

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
import re
from sklearn.utils import shuffle
import string
import matplotlib.dates as mdates
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import itertools

from PIL import Image
import PIL

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


import PIL as image_lib

from tensorflow.keras.layers import Flatten
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
# warnings.filterwarnings('ignore')
# rcParams['figure.figsize'] = 21, 9

# COMMAND ----------

!pip install yake

# COMMAND ----------

!pip install --upgrade pip

# COMMAND ----------

!pip install tensorflow_hub

# COMMAND ----------

!pip install bert 

# COMMAND ----------

!pip install opencv-python 

# COMMAND ----------

!pip install --upgrade bert 

# COMMAND ----------

!pip install tokenization 

# COMMAND ----------

!pip install wordcloud

# COMMAND ----------

!pip install bert-tensorflow 

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
import yake
import re
import urllib.request
import base64
import json
from urllib.parse import urlencode
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import tensorflow as tf
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score

try:
    %tensorflow_version 2.x
except Exception:
    pass
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import applications
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.regularizers import l2, l1
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import bert
import numpy as np 
import pandas as pd
import re
import glob
import os
import cv2
import sys
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# import plotly.express as px
import plotly.graph_objects as go
import os

# from tensorflow.keras.applications import InceptionV3
from transformers import BertTokenizer



# COMMAND ----------

rcParams['figure.figsize'] = 8, 6

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# COMMAND ----------

# MAGIC %md
# MAGIC # Dataset Preparations

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## AML `ITEMS` Dataset

# COMMAND ----------

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df1 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv"), header=True) 
df_items2021 = df1.toPandas()
df2 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv"), header=True) 
df_items2022 = df2.toPandas()

# COMMAND ----------

df_items_raw = pd.concat([df_items2021,df_items2022])
df_items_raw = df_items_raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_items_raw.iloc[583:584,] = df_items_raw.iloc[583:584,].shift(periods=2, axis=1)
df_items_raw.iloc[582:584,3:] = df_items_raw.iloc[582:584,3:].shift(periods=-1)
df_items_raw = df_items_raw.drop(df_items_raw.index[583]).reset_index().drop('index',axis=1)

df_items_raw.iloc[1043:1044,]['TITLE'] = df_items_raw.iloc[1043:1044,]['TITLE'] + ', ' + df_items_raw.iloc[1043:1044,]['COST']
df_items_raw.iloc[1043:1044,]['TITLE'] = df_items_raw.iloc[1043:1044,]['TITLE'].str.strip('"').str.replace('""','"')
df_items_raw.iloc[1043:1044,3:] = df_items_raw.iloc[1043:1044,3:].shift(-1,axis=1)

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

# MAGIC %md
# MAGIC
# MAGIC ## AML `RECOMMENDATIONS` Dataset

# COMMAND ----------

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df3 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_2021_2+BOOKS.csv"), header=True) 
df_trans2021 = df3.toPandas()
df4 = spark.read.format("csv").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_2022_2+BOOKS.csv"), header=True)
df_trans2022 = df4.toPandas()

df_trans2021 = df_trans2021.rename({'PRODUCT':'PRODUCT_ID'},axis=1)
df_recom_raw = pd.concat([df_trans2021,df_trans2022])
df_recom_raw = df_recom_raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_recom_raw = df_recom_raw.reset_index().reset_index().drop('index',axis=1)

df_recom_raw.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Cleansing

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 1. Fix data shifting issues

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

df_reomm_p = df_recom

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### 2. Transform dataset

# COMMAND ----------

def clean_recomm_df(df: pd.DataFrame) -> pd.DataFrame:

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

df_reomm_p_cleaned = clean_recomm_df(df = df_reomm_p)

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Product Data

# COMMAND ----------

df_items.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Constructing analytics dataframes

# COMMAND ----------


#Grouping product data by category
product_category = df_items.groupby(['PRD_CATEGORY']).sum()

#Removing items with zero costs for profit margins calculation by category
product_data_excost = df_items[df_items['COST']!= 0]
product_category_mean = product_data_excost.groupby(['PRD_CATEGORY']).mean()
product_category_mean['GROSS_MGN'] = (product_category_mean['PRICE'] - product_category_mean['COST']) / product_category_mean['PRICE']

#Grouping product data by publisher
product_publisher = df_items.groupby(['PUBLISHER']).sum()

#Removing items with zero costs for profit margins calculation by publisher
product_publisher_mean = product_data_excost.groupby(['PUBLISHER']).mean()
product_publisher_mean['GROSS_MGN'] = (product_publisher_mean['PRICE'] - product_publisher_mean['COST']) / product_publisher_mean['PRICE']



# COMMAND ----------

#Product Margin Heatmap with publisher against category
product_data_excost['GROSS_MGN'] = (product_data_excost['PRICE'] - product_data_excost['COST'])/product_data_excost['PRICE']
product_margin_heatmap = pd.pivot_table(product_data_excost, values = 'GROSS_MGN', index = ['PUBLISHER'], columns = ['PRD_CATEGORY'], aggfunc = np.mean)

#Keyword Extractor of titles powered by Yake
kw_extractor = yake.KeywordExtractor(top=1, stopwords=None)
product_data = df_items
product_data['Keyword'] = product_data['TITLE'].apply(lambda x: kw_extractor.extract_keywords(x))
product_data['Keyword'] = product_data['Keyword'].str[0] 
product_data['Keyword'] = product_data['Keyword'].str[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analytics Graphs

# COMMAND ----------

#Sales and Purchase by category
plt.figure(figsize=(20, 5))
width = 0.2
x = np.arange(len(product_category.index.unique()))
plt.bar(x - 0.2, product_category['QTY_SALES'], width, color='orange')
plt.bar(x + 0.2, product_category['QTY_PURCHASE'], width, color='blue')
plt.xticks(x,product_category.index.unique(), rotation=90)
plt.legend(['Sales', 'Purchases'])
plt.title("Book purchases and sales by category")
plt.show

# COMMAND ----------

#Sales and Purchase by publisher
plt.figure(figsize=(20, 5))
width = 0.2
x = np.arange(len(product_publisher.index.unique()))
plt.bar(x - 0.2, product_publisher['QTY_SALES'], width, color='orange')
plt.bar(x + 0.2, product_publisher['QTY_PURCHASE'], width, color='blue')
plt.xticks(x,product_publisher.index.unique(), rotation=90)
plt.legend(['Sales', 'Purchases'])
plt.title("Book purchases and sales by publisher")
plt.show

# COMMAND ----------

plt.figure(figsize=(20, 5))
plt.bar(product_category_mean.index, product_category_mean['GROSS_MGN'])
plt.title("Profit Margins by category")
plt.xticks(rotation=90)

# COMMAND ----------

#Profit Margin Heatmap
fig, ax = plt.subplots(figsize=(20, 20))
plt.title('Profit Margin Heatmap of publisher vs categories')
sns.heatmap(product_margin_heatmap, vmin = 0.6, vmax = 1)

# COMMAND ----------

#Word Cloud
comment_words = ''
stopwords = set(STOPWORDS)
 
# iterate through the dataframe
for val in product_data['Keyword']:

    # typecaste each val to string
    val = str(val)
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "

#Databricks generating error in below, displaying image instead which words in JupyterNotebook.

# wordcloud = WordCloud(width = 800, height = 800,
#             background_color ='white',
#             stopwords = stopwords,
#             min_font_size = 10).generate(comment_words)

# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 0)
 
# plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ![test image](files/tables/WordCloud.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transaction Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outlier Detection

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 1. Dataset Explorations

# COMMAND ----------

# DBTITLE 1,Top 50 Product Titles grouped by Quantity
df_reomm_p_cleaned_group_qty = df_reomm_p_cleaned.groupby('TITLE')['QUANTITY'].sum().reset_index().sort_values('QUANTITY', ascending=False)

# only consider top N books by count
n = 50
df_recom_qty_top_n = df_reomm_p_cleaned_group_qty.head(n)

# create a horizontal bar chart with seaborn
sns.set_style('whitegrid')
colors = ['coral' if ( x == 'Group Cash Coupon - $100' ) else 'skyblue' for x in df_recom_qty_top_n['TITLE']]
sns.barplot(x='QUANTITY', y='TITLE', data=df_recom_qty_top_n, palette=colors)

# set the chart title and axis labels
plt.title(f'Top {n} Product Titles by Quantity')
plt.xlabel('Quantity')
plt.ylabel('Product Title')

# display the chart
plt.show()

# COMMAND ----------

# DBTITLE 1,Top 50 Product Titles grouped by Amount
df_reomm_p_cleaned_group_amt = df_reomm_p_cleaned.groupby('TITLE')['AMOUNT'].sum().reset_index().sort_values('AMOUNT', ascending=False)

# only consider top N books by count
n = 50
df_recom_amt_top_n = df_reomm_p_cleaned_group_amt.head(n)

# create a horizontal bar chart with seaborn
sns.set_style('whitegrid')
colors = ['coral' if ( x == 'Group Cash Coupon - $100' ) else 'skyblue' for x in df_recom_amt_top_n['TITLE']]
sns.barplot(x='AMOUNT', y='TITLE', data=df_recom_amt_top_n, palette=colors)

# set the chart title and axis labels
plt.rcParams.update({'font.size': 22})
plt.title(f'Top {n} Product Titles by Amount')
plt.xlabel('Amount')
plt.ylabel('Product Title')

# display the chart
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 2. Remove Outliers
# MAGIC
# MAGIC - 1. remove 'Group Cash Coupon - $100
# MAGIC - 2. group data to sum quantity by ISBN13

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon = df_reomm_p_cleaned[df_reomm_p_cleaned['TITLE'] != "Group Cash Coupon - $100"]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Finalize Dataset

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon.groupby("ISBN13").sum().reset_index()
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon_stat_by_isbn.sort_values(by='QUANTITY', ascending=False)

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon_stat_by_isbn.info()

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon_stat_by_isbn.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Google API Dataset

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Load Datasets

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm_20230421_2")
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_4.json").createOrReplaceTempView("isbn_google_reomm_20230421_4")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Transform Datasets

# COMMAND ----------

googleapi = spark.sql("""
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
        description,
        lower(authors[0]) as authors,
        lower(publisher) as publisher,
        lower(categories[0]) as categories,
        lower(title) as title,
        infoLink
    FROM
        unioned
    )
    select 
    isbn,
    description,
    authors,
    publisher,
    categories,
    thumbnail,
    title,
    infoLink
    from 
    isbn_image_link_pair
    where 
    thumbnail is not null and description is not null 
""").toPandas()

# COMMAND ----------

googleapi.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Prediction Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Series Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC ### Arima Model

# COMMAND ----------

amount_over_time = df_reomm_p_cleaned_wo_coupon[[
    "year",
    "month",
    "day",
    "AMOUNT"
]].groupby([
    "year",
    "month",
    "day",
]).sum().reset_index()
amount_over_time['date'] = pd.to_datetime(amount_over_time[['year', 'month', 'day']])
amount_over_time.set_index('date', inplace=True)

# COMMAND ----------

# Split the dataset into train and test sets
train_size = int(len(amount_over_time) * 0.8)
train, test = amount_over_time[:train_size], amount_over_time[train_size:]

# Define the orders for the ARIMA models
orders = [
        (1, 1, 0), 
        (1, 1, 1), 
        (2, 1, 0), 
        (2, 1, 1)
    ]

# Train and evaluate each model
for order in orders:
    # Create the ARIMA model
    model = sm.tsa.ARIMA(train['AMOUNT'], order=order)

    # Fit the model
    results = model.fit()

    # Make predictions on the test set
    predictions = results.forecast(steps=len(test))[0]

    # Evaluate the model using the mean squared error (MSE)
    mse = mean_squared_error(test['AMOUNT'], predictions)

    # Print the order and the MSE
    print(f"ARIMA{order} MSE: {mse:.2f}")

    # Print the summary of the model
    print(results.summary())

    # Plot the residuals
    fig, ax = plt.subplots()
    ax.plot(results.resid)
    ax.set_title(f'Residuals of ARIMA{order} Model')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual')
    plt.show()

    # Plot the predictions and the actual values
    plt.plot(test.index, test['AMOUNT'], label='Actual')
    plt.plot(test.index, predictions, label='Predicted')
    plt.title(f"ARIMA{order} Predictions")
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.show()

# COMMAND ----------

# Define the range of orders to search over
p = range(0, 3)
d = range(0, 2)
q = range(0, 3)
orders = list(itertools.product(p, d, q))

# Fit and evaluate each model
best_order = None
best_mse = float('inf')

for order in orders:
    try:
        print(f"running ARIMA{order} ... ")
        # Fit the ARIMA model
        model = sm.tsa.ARIMA(train['AMOUNT'], order=order)
        model_fit = model.fit()
        
        # Make predictions on the test set
        predictions = model_fit.forecast(steps=len(test))[0]
        
        # Evaluate the model using mean squared error (MSE)
        mse = mean_squared_error(test['AMOUNT'], predictions)
        
        # Check if this is the best model so far
        if mse < best_mse:
            best_mse = mse
            best_order = order
            
    except:
        continue
        
print(f"Best order: {best_order}, Best MSE: {best_mse:.2f}")

# COMMAND ----------

# Create the ARIMA model
model = sm.tsa.ARIMA(amount_over_time['AMOUNT'], order=best_order)

# Fit the model
results = model.fit()

# Make predictions on the test set
predictions = results.forecast(steps=len(test))[0]

# Evaluate the model using the mean squared error (MSE)
mse = mean_squared_error(test['AMOUNT'], predictions)

# Print the order and the MSE
print(f"ARIMA{order} MSE: {mse:.2f}")

# Print the summary of the model
print(results.summary())

# Plot the residuals
fig, ax = plt.subplots()
ax.plot(results.resid)
ax.set_title(f'Residuals of ARIMA{best_order} Model')
ax.set_xlabel('Date')
ax.set_ylabel('Residual')
plt.show()

# Plot the predictions and the actual values
plt.plot(test.index, test['AMOUNT'], label='Actual')
plt.plot(test.index, predictions, label='Predicted')
plt.title(f"ARIMA{best_order} Predictions")
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LTSM Model

# COMMAND ----------

amount_over_time['date'] = pd.to_datetime(amount_over_time[['year', 'month', 'day']])
df_lstm = amount_over_time[["date", "AMOUNT"]]

# add 1 feature: get the difference in sales compared to the previous month 
df_lstm_diff = df_lstm.copy()

#add previous sales to the next row
df_lstm_diff['prev_AMOUNT'] = df_lstm_diff['AMOUNT'].shift(1)

#drop the null values and calculate the difference
df_lstm_diff = df_lstm_diff.dropna()
df_lstm_diff['diff'] = (df_lstm_diff['AMOUNT'] - df_lstm_diff['prev_AMOUNT'])

# #create dataframe for transformation from time series to supervised
df_lstm_supervised = df_lstm_diff.drop(['prev_AMOUNT'],axis=1)

#add 90 day lag features
for inc in range(1,90):
    field_name = 'lag_' + str(inc)
    df_lstm_supervised[field_name] = df_lstm_supervised['diff'].shift(inc)

#drop null values
df_lstm_supervised = df_lstm_supervised.dropna().reset_index(drop=True)

# COMMAND ----------

#Feature Evaluation

# Define the regression formula
formula = 'diff ~ ' + ' + '.join(['lag_' + str(i) for i in range(1, 90)])
model = smf.ols(formula=formula, data=df_lstm_supervised)

# Fit the regression
model_fit = model.fit()

# Extract the adjusted R-squared of lag_1 ~ lag_89 fields
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

# COMMAND ----------

#Data Setup

df_lstm_model = df_lstm_supervised.drop(['AMOUNT','date'],axis=1)#split train and test set
predict_test_size = 90
train_set, test_set = df_lstm_model[0:-predict_test_size].values, df_lstm_model[-predict_test_size:].values

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

#Segregating variables
X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# COMMAND ----------

#Model Setup

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False)

# COMMAND ----------

# Plot the training loss
plt.plot(history.history['loss'])
# plt.plot(validation_data)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# COMMAND ----------

# Print the model summary
model.summary()

# COMMAND ----------

y_pred = model.predict(X_test,batch_size=1) 
#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    #print(np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_lstm[-91:].date)
act_AMOUNT = list(df_lstm[-91:].AMOUNT)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_AMOUNT[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)


# COMMAND ----------

df_lstm_plot = df_lstm.drop(['date'], axis=1)
#merge with actual AMOUNT dataframe

df_AMOUNT_pred = pd.merge(df_lstm_plot,df_result,on='date',how = 'left')#plot actual and predicted

# plt.figure(figsize=(12, 6))

# Plot actual and predicted values
plt.plot(df_AMOUNT_pred['date'], df_AMOUNT_pred['AMOUNT'], label='Actual')
plt.plot(df_AMOUNT_pred['date'], df_AMOUNT_pred['pred_value'], label='Predicted')
plt.title('AMOUNT Prediction')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prophet Model

# COMMAND ----------

df_prophet_30 = df_lstm.copy()
df_prophet_30 = df_prophet_30.rename(columns={'date': 'ds', 'AMOUNT':'y'})

m_30 = Prophet(yearly_seasonality=True, daily_seasonality=True)
m_30.fit(df_prophet_30)
future_30 = m_30.make_future_dataframe(periods=30)
# future_30.tail()

forecast_30 = m_30.predict(future_30)
# forecast_30[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

plot_plotly(m_30, forecast_30)

# COMMAND ----------

# instantiate the model and fit the timeseries
prophet = Prophet(weekly_seasonality=False, changepoint_range=1,changepoint_prior_scale=0.75)
prophet.fit(df_prophet_30)

# create a future data frame 
future_30_f = prophet.make_future_dataframe(periods=30)
forecast_30_f = prophet.predict(future_30_f)

# display the most critical output columns from the forecast
# forecast_30_f[['ds','yhat','yhat_lower','yhat_upper']].head()

# plot
# fig = prophet.plot(forecast_30_f)

# plot the forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast_30_f['ds'], forecast_30_f['yhat'], label='Forecast')
ax.fill_between(forecast_30_f['ds'], forecast_30_f['yhat_lower'], forecast_30_f['yhat_upper'], alpha=0.3, label='Confidence Interval')
ax.plot(df_prophet_30['ds'], df_prophet_30['y'], label='Actual')
ax.legend()
ax.set_title('2023 Forecast (30 future days)')
plt.show()

# COMMAND ----------

df_prophet_90 = df_lstm.copy()
df_prophet_90 = df_prophet_90.rename(columns={'date': 'ds', 'AMOUNT':'y'})

m_90 = Prophet(yearly_seasonality=True, daily_seasonality=True)
m_90.fit(df_prophet_90)
future_90 = m_90.make_future_dataframe(periods=90)
# future_30.tail()

forecast_90 = m_90.predict(future_90)
# forecast_30[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

plot_plotly(m_90, forecast_90)

# COMMAND ----------

# instantiate the model and fit the timeseries
prophet = Prophet(weekly_seasonality=False, changepoint_range=1,changepoint_prior_scale=0.75)
prophet.fit(df_prophet_90)

# create a future data frame 
future_90_f = prophet.make_future_dataframe(periods=90)
forecast_90_f = prophet.predict(future_90_f)

# display the most critical output columns from the forecast
# forecast_30_f[['ds','yhat','yhat_lower','yhat_upper']].head()

# plot
# fig = prophet.plot(forecast_30_f)

# plot the forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast_90_f['ds'], forecast_90_f['yhat'], label='Forecast')
ax.fill_between(forecast_90_f['ds'], forecast_90_f['yhat_lower'], forecast_90_f['yhat_upper'], alpha=0.3, label='Confidence Interval')
ax.plot(df_prophet_90['ds'], df_prophet_90['y'], label='Actual')
ax.legend()
ax.set_title('2023 Forecast (90 future days)')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Book Descriptions Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare dataset
# MAGIC 1. Extract description and ISBN from Google API dataset
# MAGIC 2. Join description text from Google API by ISBN with 'RECOMMENDATIONS' datasets
# MAGIC 3. Set 65% percentile to define high and low sales volume
# MAGIC 4. Create target column - 'is_high_sales_volume'

# COMMAND ----------

isbn_desc = googleapi[['isbn','description']]
display(isbn_desc)

# COMMAND ----------

# DBTITLE 1,Join Google API dataset with AML `RECOMMENDATIONS` dataset and set target variable
df_text_high_sales_label = pd.merge(isbn_desc, df_reomm_p_cleaned_wo_coupon_stat_by_isbn, left_on='isbn', right_on='ISBN13', how='inner')
q_65 = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].quantile(0.65)
df_text_high_sales_label['is_high_sales_volume'] = df_text_high_sales_label['QUANTITY'].apply(lambda x: 1 if x >= q_65 else 0)

df_text_high_sales_label

# COMMAND ----------

df_text_high_sales_label = df_text_high_sales_label[['description', 'is_high_sales_volume']]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Construct Prediction Model

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 1. Input tokenized description
# MAGIC
# MAGIC   - input tokenized `description` 
# MAGIC   - label or predicted variable `is_high_sales_volume`

# COMMAND ----------

df_text = df_text_high_sales_label

max_words = 1000
max_len = 150

tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(df_text['description'].values)
X_text = tokenizer.texts_to_sequences(df_text['description'].values)
X_text = pad_sequences(X, maxlen=max_len)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 2. Split Training and Testing Datasets

# COMMAND ----------

y_text = df_text['is_high_sales_volume'].values
X_train, X_test, y_train, y_test = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 3. Model training

# COMMAND ----------

model = Sequential()
model.add(Dense(64, input_dim=max_len, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


# COMMAND ----------

history_text = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Model Evaluation

# COMMAND ----------

score = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

# DBTITLE 1,Plot accuracy over epochs
plt.plot(history_text.history['accuracy'])
plt.plot(history_text.history['val_accuracy'])
plt.title('Model Accuracy of Book Description Prediction')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# COMMAND ----------

# DBTITLE 1,Plot loss over epochs
plt.plot(history_text.history['loss'])
plt.plot(history_text.history['val_loss'])
plt.title('Model Loss of Book Description Prediction')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# COMMAND ----------

# DBTITLE 1,Accuracy Evaluation
# Make predictions on test data
y_pred = model.predict(X_test)

# Convert probabilities to predicted classes
y_pred_classes = np.round(y_pred)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)

print(accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3-class Book Cover Images Predictions
# MAGIC 3-class image classifier with Resnet50 pretrained layers

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Prepare Dataset

# COMMAND ----------

df_reomm_p_cleaned['Date'] = df_reomm_p_cleaned['year'] +'-'+ df_reomm_p_cleaned['month']+'-'+ df_reomm_p_cleaned['day']
df_reomm_p_cleaned_img = df_reomm_p_cleaned.drop_duplicates()

df_reomm_p_cleaned_img.head()

# COMMAND ----------

df_isbn_google_reomm_cleaned = googleapi.drop_duplicates()
df_isbn_google_reomm_cleaned.head()

# COMMAND ----------

df_Grouped_recom_ISBN = df_reomm_p_cleaned_img.groupby(
                    ['ISBN13']
                    ).agg(
                        {
                            'QUANTITY':sum    # Sum sales quantity
                        
                        }
                    ).reset_index(drop=False)

# COMMAND ----------

x = df_Grouped_recom_ISBN['QUANTITY'].quantile(q=0.6)
y = df_Grouped_recom_ISBN['QUANTITY'].quantile(q=0.9)

# COMMAND ----------

df_Grouped_recom_ISBN['Flag'] = np.nan

# COMMAND ----------

for i in range(0, len(df_Grouped_recom_ISBN)):
    # print(i)
    if df_Grouped_recom_ISBN['QUANTITY'].iloc[i] >= y:
        df_Grouped_recom_ISBN['Flag'].iloc[i] = 'TopSales'
    elif df_Grouped_recom_ISBN['QUANTITY'].iloc[i] >= x and df_Grouped_recom_ISBN['QUANTITY'].iloc[0] < y:
        df_Grouped_recom_ISBN['Flag'].iloc[i] = 'MediumSales'
    elif df_Grouped_recom_ISBN['QUANTITY'].iloc[i] < x:
        df_Grouped_recom_ISBN['Flag'].iloc[i] = 'LowSales'

# COMMAND ----------

df_Grouped_recom_ISBN.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Construct Prediction Model

# COMMAND ----------

# MAGIC %md define :
# MAGIC   - image size
# MAGIC   - path for the input images
# MAGIC   - train test split proportion

# COMMAND ----------

img_height,img_width=190,128
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class_Subset/",
  validation_split=0.2,
  subset="training",
  seed=123,

label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class_Subset/",
  validation_split=0.2,
  subset="validation",
  seed=123,
label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

resnet_model = Sequential()

pretrained_model_for_demo= tf.keras.applications.ResNet50(include_top=False,

                   input_shape=(190,128,3),

                   pooling='avg',classes=3,

                   weights='imagenet'
                   #weights= None
                   )

for each_layer in pretrained_model_for_demo.layers:
        each_layer.trainable=False

resnet_model.add(pretrained_model_for_demo)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(3, activation='softmax'))

# COMMAND ----------

resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

history_img = resnet_model.fit(train_ds, validation_data=validation_ds, epochs=10)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Model Evaluation

# COMMAND ----------

import matplotlib.pyplot as plotter_lib

epochs_range= range(10)

plotter_lib.plot(epochs_range, history_img.history['accuracy'], label="Training Accuracy")
plotter_lib.plot(epochs_range, history_img.history['val_accuracy'], label="Validation Accuracy")
plotter_lib.axis(ymin=0.1,ymax=1)
plotter_lib.grid()
plotter_lib.title('Model Accuracy of 3-class Book Cover Image Prediction')
plotter_lib.ylabel('Accuracy')
plotter_lib.xlabel('Epochs')
plotter_lib.legend(['train', 'validation'])

# COMMAND ----------

import matplotlib.pyplot as plt
plt.plot(history_img.history['loss'])
plt.plot(history_img.history['val_loss'])
plt.grid()
plt.title('Model Loss of 3-class Book Cover Image Prediction')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2-class Book Descriptions and Book Cover Images Prediction
# MAGIC 2-class image classifier with Binary ANN Keras Classifier Model

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Prepare Dataset

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon_stat_by_isbn['is_high_sales_volume'] = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].apply(lambda x: 1 if x >= q_65 else 0)

# COMMAND ----------

df_text_image_dataset_full = pd.merge(isbn_desc, df_reomm_p_cleaned_wo_coupon_stat_by_isbn, left_on='isbn', right_on='ISBN13', how='inner')

df_text_image_dataset_full.head()

# COMMAND ----------

# DBTITLE 1,Add '.jpg' for query image links
df_text_image_dataset = df_text_image_dataset_full[['isbn', 'description', 'is_high_sales_volume']]
df_text_image_dataset['isbn'] = df_text_image_dataset['isbn'].apply(lambda x: x + ".jpg")

# COMMAND ----------

df_text_image_dataset.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Construct Prediction Model

# COMMAND ----------

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# COMMAND ----------

# DBTITLE 1,function to pre-process the image dataset before training
# Define function to load image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

# COMMAND ----------

# DBTITLE 1,select `18000`-rows dataset with `description` and image for training
class_0_samples = df_text_image_dataset[df_text_image_dataset['is_high_sales_volume'] == 0].sample(n=9000, random_state=42)
class_1_samples = df_text_image_dataset[df_text_image_dataset['is_high_sales_volume'] == 1].sample(n=9000, random_state=42)
df = pd.concat([class_0_samples, class_1_samples])

df['isbn'] = df['isbn'].astype(str)
df['is_high_sales_volume'] = df['is_high_sales_volume'].astype(str)

# train dataset is used on below code. this code is not the true train_test_split !!!! 
train, test = train_test_split(df, test_size=0.001, random_state=42)

# COMMAND ----------


# Load images and preprocess text for training data
train_images = []
train_texts = []
train_labels = []
for index, row in train.iterrows():
    # Load image
    image_path = row['isbn']
    img = load_image('/dbfs/team_j/image_dataset/' +image_path)
    train_images.append(img)

    # Preprocess text
    text = row['description']
    
    encoded_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='tf')
    train_texts.append(encoded_text['input_ids'])

    # Get label
    label = row['is_high_sales_volume']
    train_labels.append(label)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_texts = np.squeeze(np.array(train_texts))

# COMMAND ----------

train_labels = np.array(train_labels).astype('float32')

# COMMAND ----------

# DBTITLE 1,define a concatenated image + text model    - VGG16 as pretrained layeres
from tensorflow.keras.applications import VGG16

# COMMAND ----------

# # Define image model with VGG16

image_model = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
image_model.trainable = False
x = image_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x) 
x = layers.Dense(256, activation='relu')(x) 
x = layers.Dense(128, activation='relu')(x) 
image_model = tf.keras.Model(inputs=image_model.input, outputs=x)

text_input = layers.Input(shape=(512,), dtype='int32')
text_embedding = layers.Embedding(len(tokenizer.get_vocab()), 128)(text_input)
text_conv1 = layers.Conv1D(128, 5, activation='relu')(text_embedding)
text_pool1 = layers.MaxPooling1D(5)(text_conv1)
text_conv2 = layers.Conv1D(128, 5, activation='relu')(text_pool1)
text_pool2 = layers.GlobalMaxPooling1D()(text_conv2)
text_dense = layers.Dense(256, activation='relu')(text_pool2)

# Concatenate image and text models
concatenated = layers.concatenate([image_model.output, text_dense])
dense1 = layers.Dense(512, activation='relu')(concatenated)
dense1 = layers.Dense(256, activation='relu')(dense1) 
output = layers.Dense(1, activation='sigmoid')(dense1)
model = tf.keras.models.Model(inputs=[image_model.input, text_input], outputs=output)


# COMMAND ----------

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up checkpoint path
checkpoint_path = "/dbfs/team_j/text_image_combined_model_roger_20230427_vgg16_cnn_for_text_epoch5_trial_2"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# train
history_2c = model.fit([train_images, train_texts], train_labels, validation_split=0.2, batch_size=32, epochs=5, callbacks=[cp_callback])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Model Evaluation

# COMMAND ----------

plt.plot(history_2c.history['accuracy'])
plt.plot(history_2c.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# COMMAND ----------

plt.plot(history_2c.history['loss'])
plt.plot(history_2c.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()