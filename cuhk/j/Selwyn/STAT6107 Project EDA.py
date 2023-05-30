# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yake
from rake_nltk import Rake
import nltk
from summa import keywords

# COMMAND ----------

pip install summa

# COMMAND ----------

#load data
path = 'C:\\Users\\Selwyn Cheng\\Desktop\MSc DSBS\\STAT6107\\'
data1 = pd.read_csv(path + 'ITEM_MATCHING_2021_BOOKS.csv')
data2 = pd.read_csv(path + 'ITEM_MATCHING_2022_BOOKS.csv')
data3 = pd.read_csv(path + 'RECOMMENDATION_2021_BOOKS.csv')
data4 = pd.read_csv(path + 'RECOMMENDATION_2022_BOOKS.csv')

# COMMAND ----------

#Merging products and transactions data

product_data = pd.concat([data1, data2])
transcation_data = pd.concat([data3, data4])

#product_data = data1.merge(data2)
#transaction_data = data3.merge(data4)
#data2.shape
#data4.info()
# data1.head(20)

# COMMAND ----------

date_cols = ['CREATE_DATE', 'BOOK_DATE']
float_cols = ['COST', 'PRICE', 'QTY_PURCHASE', 'QTY_SALES', 'QTY_STOCK'] #'BOOK_ORGPR', 'BOOK_PAGES'
int_cols = ['PRODUCT_ID','ISBN13']
#long_cols = []
string_cols = ['TITLE', 'VENDOR', 'BRAND', 'PRD_CATEGORY', 'PRD_ORIGIN', 'PUBLISHER', 'AUTHOR', 'TRANSLATOR', 'BOOK_COVER']

# COMMAND ----------

product_data.tail()

# COMMAND ----------

product_data = product_data.loc[~product_data['PRODUCT_ID'].str.contains('"')]
product_data = product_data.loc[~product_data['QTY_PURCHASE'].isna()]
product_data['PRODUCT_ID'] = product_data['PRODUCT_ID'].map({'C062                ': '9781920926069', 'C755                ': '9781920926755'}).fillna(product_data['PRODUCT_ID'])

# COMMAND ----------

for col in float_cols:
    product_data[col] = product_data[col].astype('float')

for col in date_cols:
    product_data[col] = pd.to_datetime(product_data[col], format = '%Y-%m-%d %H:%M:%S.%f')
    
for col in string_cols:
    product_data[col] = product_data[col].astype('str')
    
for col in int_cols:
    product_data[col] = product_data[col].astype('int64')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis

# COMMAND ----------

product_data.head()

# COMMAND ----------

product_data.info()

#na handling
#PRODUCT_ID - ''


# COMMAND ----------

product_category = product_data.groupby(['PRD_CATEGORY']).sum()
product_category
# plt.bar(product_category['PRD_CATEGORY'], product_category['QTY_SALES'])

# COMMAND ----------

len(product_category.index.unique())

# COMMAND ----------

#Sales and Purchase by category
width = 0.2
x = np.arange(len(product_category.index.unique()))
plt.bar(x - 0.2, product_category['QTY_SALES'], width, color='orange')
plt.bar(x + 0.2, product_category['QTY_PURCHASE'], width, color='blue')
plt.xticks(x,product_category.index.unique(), rotation=90)
plt.show

# COMMAND ----------

#profit margin by category
product_data_excost = product_data[product_data['COST']!= 0]
product_category_mean = product_data_excost.groupby(['PRD_CATEGORY']).mean()
product_category_mean['GROSS_MGN'] = (product_category_mean['PRICE'] - product_category_mean['COST']) / product_category_mean['PRICE']
#top 10 highest margin, date

product_category_mean

# COMMAND ----------

plt.bar(product_category_mean.index.unique(), product_category_mean['GROSS_MGN'])
plt.xticks(rotation=90)

# COMMAND ----------

publisher = product_data.groupby(['PUBLISHER']).sum()

publisher_mean = product_data.groupby(['PUBLISHER']).mean()
publisher_mean['GROSS_MGN'] = (publisher_mean['PRICE'] - publisher_mean['COST']) / publisher_mean['PRICE']

# COMMAND ----------

len(publisher.index.unique())

# COMMAND ----------

width = 0.2
x = np.arange(len(publisher.index.unique()))
plt.bar(x - 0.2, publisher['QTY_SALES'], width, color='orange')
plt.bar(x + 0.2, publisher['QTY_PURCHASE'], width, color='blue')
plt.xticks(x,publisher.index.unique(), rotation=90)
plt.show

# COMMAND ----------

#top 10 by sales
product_data.nlargest(n = 10, columns=['QTY_SALES'])


# COMMAND ----------

#top 10 by margin

product_data_excost['GROSS_MGN'] = (product_data_excost['PRICE'] - product_data_excost['COST'])/product_data_excost['PRICE']
product_data_excost.nlargest(n = 10, columns=['GROSS_MGN'])
#matrix on profit margin


# COMMAND ----------

product_sales_heatmap = pd.pivot_table(product_data, values = 'QTY_SALES', index = ['PUBLISHER'], columns = ['PRD_CATEGORY'], aggfunc = np.sum)
product_sales_heatmap

# COMMAND ----------

sns.heatmap(product_sales_heatmap)

# COMMAND ----------

product_margin_heatmap = pd.pivot_table(product_data_excost, values = 'GROSS_MGN', index = ['PUBLISHER'], columns = ['PRD_CATEGORY'], aggfunc = np.mean)
product_margin_heatmap

# COMMAND ----------

sns.heatmap(product_margin_heatmap)

# COMMAND ----------

kw_extractor = yake.KeywordExtractor(top=5, stopwords=None)
kkeywords = kw_extractor.extract_keywords("1000-piece Jigsaw Puzzle: Leopard ")

# COMMAND ----------

kkeywords

# COMMAND ----------

r = Rake()

# COMMAND ----------

r.extract_keywords_from_text("My First Puzzle Bk: Dinosaurs ")

# COMMAND ----------

r.get_ranked_phrases_with_scores()

# COMMAND ----------

TR_keywords = keywords.keywords("1000-piece Jigsaw Puzzle: Leopard ", scores=True)

# COMMAND ----------

print(TR_keywords[0:10])