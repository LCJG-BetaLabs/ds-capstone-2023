# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 0. Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - **attribute.csv:** The attribute table for Lane Crawford products. 
# MAGIC
# MAGIC - Columns:
# MAGIC
# MAGIC | Column Name         | Data Type        | Description                                                                                                    |
# MAGIC |---------------------|-----------------|----------------------------------------------------------------------------------------------------------------|
# MAGIC | atg_code            | str (primary key)| ATG code of the product.                                                                                       |
# MAGIC | prod_desc_eng       | str             | Product name in English.                                                                                       |
# MAGIC | brand_group_desc    | str             | Brand group description.                                                                                       |
# MAGIC | brand_desc          | str             | Brand description.                                                                                              |
# MAGIC | atg_bu_desc         | str             | BU description from ATG.                                                                                        |
# MAGIC | atg_class_desc      | array of str     | Class description from ATG. Combining 3 levels of class description into an array.                              |
# MAGIC | atg_subclass_desc   | str             | Subclass description from ATG. Combining 4 levels of subclass description into an array.                       |
# MAGIC | color_desc          | str             | Color description of the product (in uppercase).                                                               |
# MAGIC | compost             | str             | The product composition, indicates which fabric has been used to make the product. Null for most products except BU in WW and MW.|
# MAGIC | care                | str             | Product Information. Same as Product Information section on LC website.                                         |
# MAGIC | size_n_fit          | str             | Size and fit information. Same as Fit & Styling section (or How to Apply section for COS) on LC website.        |
# MAGIC | long_desc           | str             | Product description. Same as Product description section on LC website.                                        |
# MAGIC | price               | float           | Product price.                                                                                                  |
# MAGIC | img_list            | array of str     | An array of image names of the product. e.g. for product AAE789, the img_list is ["AAE789_1_xl.jpg", "AAE789_bk_xl.jpg", "AAE789_in_xl.jpg", "AAE789_ro_xl.jpg"] |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - Set up DBFS

# COMMAND ----------

import os
import tensorflow.compat.v1 as tf
 
team_container = "capstone2023-hku-team-a"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
 
glove_path = "glove"
save_glove_path = "/dbfs/glove"
dbutils.fs.cp(os.path.join(team_path, glove_path), save_glove_path, recurse=True) 

# COMMAND ----------

dbutils.fs.ls(glove_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1. Import Data

# COMMAND ----------

# import farfetch data
import pandas as pd

team_acount_name = "capstone2023hku"
file_path1 = "farfetch.csv"

file_url1 = f"https://{team_acount_name}.blob.core.windows.net/{team_container}/{file_path1}?st=2023-03-09&se=2023-05-31&spr=https&sv=2021-06-08&si=hku-team-a&sr=c&sig=IAR5rxam3Qc502Z6Ar3p7m7pHPb4ptOp2MRa8rFFvNI%3D"

# Load data
farfetch_df = pd.read_csv(file_url1)
farfetch_df.head()

# COMMAND ----------

print(len(farfetch_df))

# COMMAND ----------

# remove duplicates
farfetch_df = farfetch_df.drop_duplicates(subset=['farfetch_id'])
print(len(farfetch_df))

# COMMAND ----------

# import Lane Crawford data
import pandas as pd

container = "data3"
acount_name = "capstone2023cuhk"
file_path = "competitor_analysis/attribute.csv"

file_url = f"https://{acount_name}.blob.core.windows.net/{container}/{file_path}?st=2023-03-09&se=2023-05-31&spr=https&sv=2021-06-08&si=student3&sr=c&sig=j5hLKIQkjzR74E/NMKPXADSPCnJy75L3S%2BaHlJuWu5k%3D"

# Load data
lc_df = pd.read_csv(file_url)
lc_df.head()

# COMMAND ----------

farfetch_df_spark =spark.createDataFrame(farfetch_df)
display(farfetch_df_spark)

# COMMAND ----------

lc_df_spark =spark.createDataFrame(lc_df)
display(lc_df_spark)

# COMMAND ----------

print(lc_df.columns)
print(farfetch_df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - Comparision
# MAGIC
# MAGIC | LC    | Farfetch       | Description of LC attribute                                                                                                  |
# MAGIC |---------------------|-----------------|----------------------------------------------------------------------------------------------------------------|
# MAGIC | atg_code            | farfetch_id | ATG code of the product.                                                                                       |
# MAGIC | prod_desc_eng       | pro_desc_eng        | Product name in English.                                                                                       |
# MAGIC | brand_group_desc    | group_desc         | Brand group description.                                                                                       |
# MAGIC | brand_desc          | brand_desc            | Brand description.                                                                                              |
# MAGIC | atg_bu_desc         |  /  | BU description from ATG.                                                                                        |
# MAGIC | atg_class_desc      |  atg_class_desc     | Class description from ATG. Combining 3 levels of class description into an array.                              |
# MAGIC | atg_subclass_desc   |     /       | Subclass description from ATG. Combining 4 levels of subclass description into an array.                       |
# MAGIC | color_desc          |  color_desc | Color description of the product (in uppercase).                                                               |
# MAGIC | compost             |  compost  | The product composition, indicates which fabric has been used to make the product. Null for most products except BU in WW and MW.|
# MAGIC | care                |   care  | Product Information. Same as Product Information section on LC website.                                         |
# MAGIC | size_n_fit          | size_n_fit | Size and fit information. Same as Fit & Styling section (or How to Apply section for COS) on LC website.        |
# MAGIC | long_desc           |  long_desc | Product description. Same as Product description section on LC website.                                        |
# MAGIC | price               |  price   | Product price.                                                                                                  |
# MAGIC | img_list            | img_list    | An array of image names of the product. e.g. for product AAE789, the img_list is ["AAE789_1_xl.jpg", "AAE789_bk_xl.jpg", "AAE789_in_xl.jpg", "AAE789_ro_xl.jpg"] |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 2. Data Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2.1 Choose attributes
# MAGIC
# MAGIC |Lane Crawford|Farfetch|
# MAGIC |---|---|
# MAGIC |atg_code|farfetch_id|
# MAGIC |prod_desc_eng|pro_desc_eng|
# MAGIC |brand_desc|brand_desc|
# MAGIC |color_desc|color_desc|
# MAGIC |compost|compost|
# MAGIC |long_desc|long_desc|

# COMMAND ----------

lc_df_clean = lc_df[['atg_code', 'prod_desc_eng', 'brand_desc', 'color_desc', 'compost', 'long_desc']]

lc_df_clean.head()

# COMMAND ----------

farfetch_df_clean = farfetch_df[['farfetch_id', 'pro_desc_eng', 'brand_desc', 'color_desc', 'compost', 'long_desc']]
farfetch_df_clean = farfetch_df_clean.rename(columns={'pro_desc_eng': 'prod_desc_eng'})

farfetch_df_clean.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2.2 Lemmatization & Tokenization

# COMMAND ----------

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def text_prepocessing(text):
    text = str(text)
    text = text.lower()
    text = text.replace('-', ' ') 
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)

    # remove stop words, lemmatization and tokenization
    lemmatizer=nltk.stem.WordNetLemmatizer()
    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')
    text = text.strip()
    tokens=wpt.tokenize(text)
    text_token = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    text_processed =' '.join(text_token)
    
    return text_processed

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### - Process each column

# COMMAND ----------

lc_df_processed = lc_df_clean

lc_df_processed['prod_desc_processed'] = lc_df_processed['prod_desc_eng'].apply(lambda X: text_prepocessing(X)) 
lc_df_processed['brand_processed'] = lc_df_processed['brand_desc'].apply(lambda X: text_prepocessing(X)) 
lc_df_processed['color_processed'] = lc_df_processed['color_desc'].apply(lambda X: text_prepocessing(X)) 
lc_df_processed['compost_processed'] = lc_df_processed['compost'].apply(lambda X: text_prepocessing(X)) 
lc_df_processed['long_desc_processed'] = lc_df_processed['long_desc'].apply(lambda X: text_prepocessing(X)) 

lc_df_processed.head()

# COMMAND ----------

farfetch_df_processed = farfetch_df_clean

farfetch_df_processed['prod_desc_processed'] = farfetch_df_processed['prod_desc_eng'].apply(lambda X: text_prepocessing(X)) 
farfetch_df_processed['brand_processed'] = farfetch_df_processed['brand_desc'].apply(lambda X: text_prepocessing(X)) 
farfetch_df_processed['color_processed'] = farfetch_df_processed['color_desc'].apply(lambda X: text_prepocessing(X)) 
farfetch_df_processed['compost_processed'] = farfetch_df_processed['compost'].apply(lambda X: text_prepocessing(X)) 
farfetch_df_processed['long_desc_processed'] = farfetch_df_processed['long_desc'].apply(lambda X: text_prepocessing(X)) 

farfetch_df_processed.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###- Show the results

# COMMAND ----------

lc_df_processed_spark =spark.createDataFrame(lc_df_processed)
display(lc_df_processed_spark)
farfetch_df_processed_spark =spark.createDataFrame(farfetch_df_processed)
display(farfetch_df_processed_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 3. NER

# COMMAND ----------

# %sh
# #/databricks/python3/bin/pip install spacy 
# /databricks/python3/bin/python3 -m spacy download en

# COMMAND ----------

# import spacy

# # Load English tokenizer, tagger, parser, NER and word vectors
# nlp = spacy.load("en_core_web_lg")

# COMMAND ----------

# def NER(text):
#     doc = nlp(text)
#     labels = []
#     for ent in doc.ents:
#         labels.append(ent.label_)
#     labels = ','.join(labels)
#     return labels

# COMMAND ----------

# lc_df_processed['labels'] = lc_df_processed['text_processed'].apply(lambda x: NER(x)) 
# farfetch_df_processed['labels'] = farfetch_df_processed['text_processed'].apply(lambda x: NER(x)) 

# COMMAND ----------

# lc_df_processed.head()

# COMMAND ----------

# farfetch_df_processed.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 4.Extract Embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### - Use Spacy pre-trained vector

# COMMAND ----------

# import spacy
# import numpy as np

# nlp = spacy.load("en_core_web_lg")

# def get_word_vectors(tokens):
#     word_vectors = []
#     for token in tokens:
#         word_vectors.append(nlp(token).vector)
#     return np.array(word_vectors)

# COMMAND ----------

# lc_df_processed['embeddings'] = lc_df_processed['text_token'].apply(lambda x: get_word_vectors(x)) 
# farfetch_df_processed['embeddings'] = farfetch_df_processed['text_token'].apply(lambda x: get_word_vectors(x)) 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### - Use Glove pre-trained model
# MAGIC
# MAGIC - Reference: https://towardsdatascience.com/nlp-building-a-summariser-68e0c19e3a93

# COMMAND ----------

# define dict to hold a word and its vector
import numpy as np

word_embeddings = {}

glove_model_path = "/dbfs/" + glove_path + "/glove.6B.100d.txt"

f = open(glove_model_path, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()# check the length

len(word_embeddings) # 400000

# COMMAND ----------

def get_vector(x):
    if len(x) != 0:
        vector = sum([word_embeddings.get(w, np.zeros((100,))) for w in x.split()])/(len(x.split())+0.001)
    else:
        vector = np.zeros((100,))
    return vector

# COMMAND ----------

lc_df_processed['prod_desc_vector'] = lc_df_processed['prod_desc_processed'].apply(lambda x: get_vector(x))
lc_df_processed['brand_vector'] = lc_df_processed['brand_processed'].apply(lambda x: get_vector(x))
lc_df_processed['color_vector'] = lc_df_processed['color_processed'].apply(lambda x: get_vector(x))
lc_df_processed['compost_vector'] = lc_df_processed['compost_processed'].apply(lambda x: get_vector(x))
lc_df_processed['long_desc_vector'] = lc_df_processed['long_desc_processed'].apply(lambda x: get_vector(x))

lc_df_processed.head()

# COMMAND ----------

farfetch_df_processed['prod_desc_vector'] = farfetch_df_processed['prod_desc_processed'].apply(lambda x: get_vector(x))
farfetch_df_processed['brand_vector'] = farfetch_df_processed['brand_processed'].apply(lambda x: get_vector(x))
farfetch_df_processed['color_vector'] = farfetch_df_processed['color_processed'].apply(lambda x: get_vector(x))
farfetch_df_processed['compost_vector'] = farfetch_df_processed['compost_processed'].apply(lambda x: get_vector(x))
farfetch_df_processed['long_desc_vector'] = farfetch_df_processed['long_desc_processed'].apply(lambda x: get_vector(x))

farfetch_df_processed.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###  *Cosine similarity approach testing

# COMMAND ----------

cv_matrix1 = lc_df_processed['word_vector'][2]
cv_matrix2 = farfetch_df_processed['word_vector'][3]

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity

cos_sim_list = []

# sample = farfetch_df_processed[:100]

for index,row in farfetch_df_processed.iterrows():
    if row['brand_desc'] == 'Acne Studios':
        f_id = row['farfetch_id']
        f_matrix = row['word_vector']
        cos_sim = cosine_similarity(cv_matrix1.reshape(1,100), f_matrix.reshape(1,100))[0,0]
        cos_sim_list.append((f_id, cos_sim))


# COMMAND ----------

com_sim_df = pd.DataFrame(cos_sim_list, columns=['f_id', 'cos_sim'])
com_sim_df

# COMMAND ----------

com_sim_df = com_sim_df.sort_values(by=['cos_sim'], ascending=False)
com_sim_df

# COMMAND ----------

(farfetch_df_processed[farfetch_df_processed['farfetch_id'] == 19849908]['text_token'])

# COMMAND ----------

from pyspark.sql import functions
farfetch_df_processed_spark.where(farfetch_df_processed_spark.farfetch_id == 13431705).display()

# COMMAND ----------

farfetch_df_processed_spark.where(farfetch_df_processed_spark.brand_desc == 'Acne Studios').display()

# COMMAND ----------

lc_df_processed_spark.where(lc_df_processed_spark.brand_desc == 'ACNE STUDIOS').display()

# COMMAND ----------

lc_df_processed['long_desc'][1]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 5. Results Saving

# COMMAND ----------

lc_df_processed.columns

# COMMAND ----------

lc_df_result = lc_df_processed[['atg_code', 
                                'prod_desc_vector', 
                                'brand_vector', 
                                'color_vector', 
                                'compost_vector',
                                'long_desc_vector']]

farfetch_df_result = farfetch_df_processed[['farfetch_id', 
                                'prod_desc_vector', 
                                'brand_vector', 
                                'color_vector', 
                                'compost_vector',
                                'long_desc_vector']]                               

# COMMAND ----------

lc_df_result.head()

# COMMAND ----------

farfetch_df_result.head()

# COMMAND ----------

dbutils.fs.mkdirs("nlp")
lc_df_result.to_csv("/dbfs/nlp/lc_result_v2.csv", index = False)
dbutils.fs.cp("nlp", os.path.join(team_path, "nlp"), recurse=True)

# COMMAND ----------

dbutils.fs.mkdirs("nlp")
lc_df_result.to_json("/dbfs/nlp/lc_result_v2.json")
dbutils.fs.cp("nlp", os.path.join(team_path, "nlp"), recurse=True)

# COMMAND ----------

dbutils.fs.mkdirs("nlp")
farfetch_df_result.to_csv("/dbfs/nlp/farfetch_result_v2.csv", index = False)
dbutils.fs.cp("nlp", os.path.join(team_path, "nlp"), recurse=True)

# COMMAND ----------

dbutils.fs.mkdirs("nlp")
farfetch_df_result.to_json("/dbfs/nlp/farfetch_result_v2.json")
dbutils.fs.cp("nlp", os.path.join(team_path, "nlp"), recurse=True)