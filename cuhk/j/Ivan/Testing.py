# Databricks notebook source
import os
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
import seaborn as sns
from shutil import copyfile
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
import re
from sklearn.utils import shuffle
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 21, 9


# COMMAND ----------

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df_items = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_*.csv"), header=True)
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 

# df_items.createOrReplaceTempView("df_items") # spark read
# df_reomm.createOrReplaceTempView("df_reomm") # spark read

df_items_p = df_items.toPandas() # padnas 
df_reomm_p = df_reomm.toPandas() # padnas 

# external dataset from googleapi books
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm")


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

# COMMAND ----------

df_reomm_p_cleaned['Date'] = df_reomm_p_cleaned['year'] +'-'+ df_reomm_p_cleaned['month']+'-'+ df_reomm_p_cleaned['day']
df_reomm_p_cleaned = df_reomm_p_cleaned.drop_duplicates()

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

# COMMAND ----------

df_reomm_p_cleaned.columns

# COMMAND ----------

df_Grouped_recom = df_reomm_p_cleaned.groupby(
                    ['Date', 'year', 'month', 'day','PRODUCT','ISBN13','TITLE','SHOP_NO']
                    ).agg(
                        {
                            'QUANTITY':sum,    # Sum sales quantity
                            'PRICE': 'first'  # get the first price per group
                        }
                    ).reset_index(drop=False)

# COMMAND ----------

df_Grouped_recom_ISBN = df_reomm_p_cleaned.groupby(
                    ['ISBN13']
                    ).agg(
                        {
                            'QUANTITY':sum    # Sum sales quantity
                        
                        }
                    ).reset_index(drop=False)

# COMMAND ----------

len(df_Grouped_recom_ISBN)

# COMMAND ----------

df_Grouped_recom_ISBN.head()

# COMMAND ----------


x = df_Grouped_recom_ISBN['QUANTITY'].quantile(q=0.6)

y = df_Grouped_recom_ISBN['QUANTITY'].quantile(q=0.9)




# COMMAND ----------

df_Grouped_recom_ISBN['Flag'] = np.nan

# COMMAND ----------

for i in range(0, len(df_Grouped_recom_ISBN)):
    print(i)
    if df_Grouped_recom_ISBN['QUANTITY'].iloc[i] >= y:
        df_Grouped_recom_ISBN['Flag'].iloc[i] = 'TopSales'
    elif df_Grouped_recom_ISBN['QUANTITY'].iloc[i] >= x and df_Grouped_recom_ISBN['QUANTITY'].iloc[0] < y:
        df_Grouped_recom_ISBN['Flag'].iloc[i] = 'MediumSales'
    elif df_Grouped_recom_ISBN['QUANTITY'].iloc[i] < x:
        df_Grouped_recom_ISBN['Flag'].iloc[i] = 'LowSales'

# COMMAND ----------

df_Grouped_recom_ISBN

# COMMAND ----------

target_dir = "./team_j_downloaded_images/image_Class/"
# Check if the directory exists
if os.path.exists(dest_dir):
    print(f'Directory {dest_dir} exists')
else:
    # Create the directory if it doesn't exist
    os.makedirs(dest_dir)
    print(f'Directory {dest_dir} created')

# COMMAND ----------

df_Grouped_recom_ISBN.Flag.unique()

# COMMAND ----------

target_dir = "./team_j_downloaded_images/image_Class"
# Check if the directory exists
if os.path.exists(target_dir):
    print(f'Directory {target_dir} exists')
else:
    # Create the directory if it doesn't exist
    os.makedirs(target_dir)
    print(f'Directory {target_dir} created')

# COMMAND ----------

image_dir = "/dbfs/team_j_downloaded_images/raw_20230423/"
raw_image_list = os.listdir(f'{image_dir}')


# COMMAND ----------

df_Grouped_recom_ISBN['ISBN13'].head()

# COMMAND ----------

f'{image_dir}{raw_image_list[0]}'

# COMMAND ----------

df_Grouped_recom_ISBN.iloc[0]

# COMMAND ----------

glob.glob(f'{image_dir}*')

# COMMAND ----------

import glob

glob.glob(f'/dbfs/team_j_downloaded_images/*')

# COMMAND ----------

 df_Grouped_recom_ISBN['Flag'].unique()

# COMMAND ----------

import os
target_dir = "/dbfs/team_j_downloaded_images/image_Class/"

os.path.exists(target_dir)

# COMMAND ----------

df_Grouped_recom_ISBN['Flag'].unique()

# COMMAND ----------

os.makedirs("/dbfs/team_j_downloaded_images/image_Class/TopSales")

# COMMAND ----------

df_Grouped_recom_ISBN['Flag'].unique()

# COMMAND ----------

for i in glob.glob('/dbfs/team_j_downloaded_images/image_Class/LowSales/*'):
    os.remove(i)

# COMMAND ----------

import glob
import os
import shutil
for i in range(0, len(df_Grouped_recom_ISBN)):
    if df_Grouped_recom_ISBN['Flag'].iloc[i] == 'TopSales':
        try:
            ISBN = df_Grouped_recom_ISBN['ISBN13'].iloc[i]

            shutil.copyfile(f'/dbfs/team_j_downloaded_images/raw_20230423/{ISBN}.jpg', f'/dbfs/team_j_downloaded_images/image_Class/TopSales/{ISBN}.jpg')
        except:
            print('No Such ISBN')
        #print(f"copied file {file_name} to file pat : {dest_file}")

# COMMAND ----------

import glob
import os
import shutil
for i in range(0, len(df_Grouped_recom_ISBN)):
    if df_Grouped_recom_ISBN['Flag'].iloc[i] == 'LowSales':
        try:
            ISBN = df_Grouped_recom_ISBN['ISBN13'].iloc[i]

            shutil.copyfile(f'/dbfs/team_j_downloaded_images/raw_20230423/{ISBN}.jpg', f'/dbfs/team_j_downloaded_images/image_Class/LowSales/{ISBN}.jpg')
        except:
            print('No Such ISBN')
        #print(f"copied file {file_name} to file pat : {dest_file}")

# COMMAND ----------

import glob
import os
import shutil
for i in range(0, len(df_Grouped_recom_ISBN)):
    if df_Grouped_recom_ISBN['Flag'].iloc[i] == 'MediumSales':
        try:
            ISBN = df_Grouped_recom_ISBN['ISBN13'].iloc[i]

            shutil.copyfile(f'/dbfs/team_j_downloaded_images/raw_20230423/{ISBN}.jpg', f'/dbfs/team_j_downloaded_images/image_Class/MediumSales/{ISBN}.jpg')
        except:
            print('No Such ISBN')
        #print(f"copied file {file_name} to file pat : {dest_file}")

# COMMAND ----------

team_j_downloaded_images/image_Class/TopSales

# COMMAND ----------

df_Grouped_recom_ISBN['Flag'].unique()

# COMMAND ----------

df_merged = pd.merge(df_Grouped_recom,df_isbn_google_reomm_cleaned, left_on = 'ISBN13', right_on = 'isbn', how = 'inner')

# COMMAND ----------

df_merged.columns


# COMMAND ----------

plt.figure(figsize=(10,4))
plt.xlim(-100, 40)
sns.boxplot(x=df_merged.QUANTITY)

# COMMAND ----------

plt.figure(figsize=(10,4))
plt.xlim(-100, 10000)
sns.boxplot(x=df_merged.PRICE)

# COMMAND ----------

df_merged.head()

# COMMAND ----------

len(df_merged.authors.unique())

# COMMAND ----------

df_merged['description'].iloc[5]

# COMMAND ----------

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# COMMAND ----------

df_merged.columns

# COMMAND ----------

df_merged = df_merged.dropna(axis=0)


# COMMAND ----------

len(df_merged)

# COMMAND ----------

# Split the data into training and testing sets
train_data, test_data = train_test_split(df_merged, test_size=0.2, random_state=42)

# COMMAND ----------

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', string)

def remove_unwanted(document):

    # remove user mentions
    document = re.sub("@[A-Za-z0-9_]+"," ", document)
    # remove URLS 
    document = re.sub(r'http\S+', ' ', document)
    # remove hashtags
    document = re.sub("#[A-Za-z0-9_]+","", document)
    # remove emoji's
    document = remove_emoji(document)
    # remove punctuation
    document = re.sub("[^0-9A-Za-z ]", "" , document)
    # remove double spaces
    document = document.replace('  ',"")
    
    return document.strip()

def remove_words(tokens):
    stopwords = nltk.corpus.stopwords.words('english') # also supports german, spanish, portuguese, and others!
    stopwords = [remove_unwanted(word) for word in stopwords] # remove puntcuation from stopwords
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    return cleaned_tokens
    
def pipeline(document, rule = 'lemmatize'):
    # first lets normalize the document
    document = normalize(document)
    # now lets remove unwanted characters
    document = remove_unwanted(document)
    # create tokens
    tokens = document.split()
    # remove unwanted words
    tokens = remove_words(tokens)
    # lemmatize or stem or 
    if rule == 'lemmatize':
        tokens = lemmatize(tokens)
    elif rule == 'stem':
        tokens = stemmer(tokens)
    else:
        print(f"{rule} Is an invalid rule. Choices are 'lemmatize' and 'stem'")
    
    return " ".join(tokens)

# COMMAND ----------

normalize = lambda document: document.lower()

lemma = WordNetLemmatizer()

def lemmatize(tokens):
    lemmatized_tokens = [lemma.lemmatize(token, pos = 'v') for token in tokens]
    return lemmatized_tokens



# COMMAND ----------

df_merged['cleaned_desc'] = df_merged['description'].apply(lambda x:pipeline(x) )

# COMMAND ----------

df_merged.head()

# COMMAND ----------

df_merged.columns

# COMMAND ----------

df_merged_sales = df_merged.groupby(
                    ['ISBN13', 'authors', 'publisher', 'categories',
                    'cleaned_desc','title']
                    ).agg(
                        {
                            'QUANTITY':sum,    # Sum sales quantity
                            'PRICE': 'first'
                        }
                    ).reset_index(drop=False)

# COMMAND ----------

df_merged_sales.sort_values(by='QUANTITY', ascending=False, inplace = True)

# COMMAND ----------

#Create a top sales flag
df_merged_sales['Top_Sales_Flag'] = df_merged_sales['QUANTITY'].apply(lambda x: 1 if x>= df_merged_sales['QUANTITY'].quantile(q=0.75) else 0 )

# COMMAND ----------

df_merged_sales.columns

# COMMAND ----------

one_hot_authors = pd.get_dummies(df_merged_sales['authors'])
one_hot_publisher = pd.get_dummies(df_merged_sales['publisher'])
one_hot_categories = pd.get_dummies(df_merged_sales['categories'])

# COMMAND ----------

df_merged_sales.head()

# COMMAND ----------

df_items_p.head()

# COMMAND ----------

df_reomm_p