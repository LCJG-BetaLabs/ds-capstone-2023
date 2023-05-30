# Databricks notebook source
# MAGIC %md
# MAGIC #### Examine data

# COMMAND ----------

import pandas as pd
import os
container = "data3"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
# copy folder from ABFS to local
dbutils.fs.cp(os.path.join(data_path, "pattern_recognition"), "file:/pattern_recognition", recurse=True) 
# Load data
df = pd.read_csv(os.path.join("file:/pattern_recognition", "attribute.csv"))

# COMMAND ----------

# View the first few rows of the DataFrame
df.head()

# COMMAND ----------

# View basic statistical information about the DataFrame
df.describe()

# COMMAND ----------

# View the names of the columns in the DataFrame
df.columns

# COMMAND ----------

# View the number of rows and columns in the DataFrame
df.shape

# COMMAND ----------

# View the data types of each column in the DataFrame
df.dtypes

# COMMAND ----------

# Check for missing values in each column
df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Text description

# COMMAND ----------

pip install nltk

# COMMAND ----------

pip install keybert

# COMMAND ----------

filePath = "/Workspace/Repos/Team B/cleaned_data.parquet"
df = spark.read.parquet(filePath)
from pyspark.sql.functions import *
df = df.withColumn("long_desc", (concat(col("prod_desc_eng"), lit(' '), col("long_desc"))))
display(df)

# COMMAND ----------

# keyword extraction
from keybert import KeyBERT
kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')
text = df.collect()[0]['long_desc']
keywords = kw_model.extract_keywords(text)

# COMMAND ----------

import numpy as np
' '.join(list(np.array(kw_model.extract_keywords(text))[0:5,0]))

# COMMAND ----------

keywords[0][0]
def extract_keyword():
    return udf(lambda x: ' '.join(list(np.array(kw_model.extract_keywords(x))[0:5,0])))

# COMMAND ----------

df.withColumn('keyword',extract_keyword()(col('long_desc'))).display()

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
from pyspark.ml.feature import Word2Vec, StopWordsRemover
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer

nltk.download('stopwords')
nltk_stopwords = stopwords.words("english")
tokenizer = Tokenizer(inputCol="long_desc", outputCol="tokens")
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_text").setStopWords(nltk_stopwords)

# COMMAND ----------

# Map document to vector of 300 in size
w2v = Word2Vec(vectorSize=10, minCount=0, inputCol="filtered_text", outputCol="features")

# Pipline and fit a doc2vec model
doc2vec_pipeline = Pipeline(stages=[tokenizer, stopwords_remover, w2v])
doc2vec_model = doc2vec_pipeline.fit(df)
doc2vecs_df = doc2vec_model.transform(df)

# Check the output
doc2vecs_df.display()

# COMMAND ----------

pattern = [['checks: modified stripes consisting of crossed horizontal and vertical lines which form squares.'],
           ['graphic_print: Pattern includes geometrical pattern, image, drawing. Floral and animal print are grouped in this category.'],
           ['multi_color:  tie-dyed, joining of different color of fabrics without word or graphic print'],
           ['plain: Items consist of one color without obvious pattern'],
           ['stripe: Items contain striped pattern of any direction and any striped thickness'],
           ['word_print: Items consist of obvious word printing without having graphic print.']]

def split():
    return udf(lambda x: x.split(':')[0])

# COMMAND ----------

from pyspark.sql.types import *
schema_1 = StructType([StructField("long_desc",StringType(),False)])
pt = spark.createDataFrame(pattern,schema=schema_1)
pt = pt.withColumn('pattern',split()(col('long_desc')))
pattern_feature = doc2vec_model.transform(pt)
pattern_feature.display()

# COMMAND ----------

# Method 1
df_transformed = doc2vecs_df
df_cross = df_transformed.select(
    col('atg_code').alias('product_id'),
    col('features').alias('vecs1')).crossJoin(pattern_feature.select(
        col('pattern').alias('pattern'),
        col('features').alias('vecs2'))
)

# COMMAND ----------

from scipy import spatial

@udf(returnType=FloatType())
def sim(x, y):
    return float(1 - spatial.distance.euclidean(x, y))

df_cross = df_cross.withColumn('sim', sim(df_cross['vecs1'], df_cross['vecs2']))

# COMMAND ----------

test_id = 'BUE685'
pdf1 = df_cross.filter(col('product_id')==test_id).select('product_id','pattern','sim').toPandas()
sim_top10 = pdf1[pdf1.sim<1].sort_values('sim', ascending=False).head(10)
sim_top10

# COMMAND ----------

def search_sim(product_id):
    pdf1 = df_cross.filter(col('product_id')==product_id).select('product_id','pattern','sim').toPandas()
    sim_top = pdf1[pdf1.sim<1].sort_values('sim', ascending=False)['pattern'].iloc[0]
    return (sim_top)
search_sim('BUE685')

# COMMAND ----------

import os
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
spark.read.parquet(os.path.join(team_path, "smote_df_with_vectors_train_resnet.parquet")).display()

# COMMAND ----------

# Databricks notebook source

import os

import pandas as pd


attr = df.toPandas()


pattern_keyword = {

    "stripe": ["stripe", "stripes", "striped"],

    "checks": ["checks", "gingham", "plaid", "tartan"],

    "plain": ["plain"],

    "graphic_print": ["graphic print", "floral", "tulip print", "print", "leopard", "cheetah", "zebra", "giraffe", "tiger", "cow", "deer"],

    "multi_color": ["tie dye", "tie&dye", "tie dyed", "tie-dye"],

    "word_print": ["logo print", "front logo", "logo"],

}

def get_pattarn_tag(text):
    for tag, keywords in pattern_keyword.items():
        if any(keyword in text for keyword in keywords):
            return tag
    return "unknown"

attr = attr.fillna("")

attr["text"] = attr[["prod_desc_eng", "size_n_fit", "long_desc", "care"]].agg(" ".join, axis=1)

attr["text"] = attr["text"].astype(str).str.lower()

attr["tag"] = attr["text"].apply(lambda x: get_pattarn_tag(x))
print(len(attr[attr["tag"]=='unknown']))
result = attr[["atg_code", "tag"]]

#result = result[result["tag"] != "unknown"]

result

# COMMAND ----------

display(result)

# COMMAND ----------

index = (result["tag"] == "unknown")
result.loc[index,'tag'] = result.loc[index,'atg_code'].apply(lambda x: search_sim(x))
result

# COMMAND ----------

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

import pandas as pd

import os

 

 

def get_image_path(root, atg_code) -> str:

    return os.path.join(root, f"{atg_code}.jpg")

 

 

def read_image(root, atg_code):

    image_path = get_image_path(root, atg_code)

    try:

        image = Image.open(image_path).convert("RGB")

        return image

    except Exception as e:

        raise Exception(f"Failed to read image {image_path}") from e

 

 

def show_set(set_atgs, tag, root):

    fig, axes = plt.subplots(1, len(set_atgs), figsize=(20, 5))

    for i, atg in enumerate(set_atgs):

        img = read_image(root, atg)

        if img:

            axes[i].imshow(img)

            axes[i].set_title(atg)

            axes[i].grid(False)

            axes[i].axis("off")

 

    fig.suptitle(f"tag: {tag}", fontsize=20)

    plt.tight_layout()

    plt.show()

root = "/dbfs/FileStore/lanecrawford_img"

tags = result["tag"].drop_duplicates().values

for tag in tags:

    df = result[result["tag"] == tag]

    if len(df) >= 10:

        atgs = np.array(df.sample(10)["atg_code"])

    else:

        atgs = np.array(df["atg_code"])

        print(f"# of items for tag '{tag}': {len(df['atg_code'])}")

    print(atgs)

    show_set(atgs, tag, root)

# COMMAND ----------

result['tag'].value_counts()

# COMMAND ----------

# Method 2
df_transformed = doc2vecs_df
df_cross = df_transformed.select(
    col('atg_code').alias('id1'),
    col('features').alias('vecs1')).crossJoin(df_transformed.select(
        col('atg_code').alias('id2'),
        col('features').alias('vecs2'))
)
df_cross = df_cross.withColumn('sim', sim(df_cross['vecs1'], df_cross['vecs2']))

# COMMAND ----------

test_id = 'BUE685'
pdf1 = df_cross.filter(col('id1')==test_id).toPandas()
sim_top10 = pdf1[pdf1.sim<1].sort_values('sim', ascending=False).head(10)
sim_top10

# COMMAND ----------

pip install opencv-python

# COMMAND ----------

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = cv2.imread("/pattern_recognition/image/BUE685_in_xl.jpg",flags=1)
#print(image)
plt.imshow(image)
df.select('*').where(col('atg_code')=='BUE685').display()

# COMMAND ----------

def show_similar_prod(code):

    path = os.path.join('/pattern_recognition/image/', f"{code}_in_xl.jpg")
    image = cv2.imread(path,flags=1)
#print(image)
    plt.imshow(image)
    df.select('*').where(col('atg_code')==code).display()
    
code = 'BWE399'
show_similar_prod(code)

# COMMAND ----------


code = 'BVY624'
show_similar_prod(code)

# COMMAND ----------


code = 'BVE863'
show_similar_prod(code)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/

# COMMAND ----------

label_df = pd.read_csv("/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/items_pattern.csv",header=0)
labels_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/items_pattern.csv")
label_df