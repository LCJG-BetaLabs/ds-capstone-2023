# Databricks notebook source
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

df.shape


# COMMAND ----------

pip install nltk

# COMMAND ----------

filePath = "/Workspace/Repos/Team B/cleaned_data.parquet"
df = spark.read.parquet(filePath)
from pyspark.sql.functions import *
df = df.withColumn("long_desc", (concat(col("prod_desc_eng"), lit(' '), col("long_desc"))))
display(df)

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

from scipy import spatial

@udf(returnType=FloatType())
def sim(x, y):
    return float(1 - spatial.distance.euclidean(x, y))

df_cross = df_cross.withColumn('sim', sim(df_cross['vecs1'], df_cross['vecs2']))

def search_sim(product_id):
    pdf1 = df_cross.filter(col('product_id')==product_id).select('product_id','pattern','sim').toPandas()
    sim_top = pdf1[pdf1.sim<1].sort_values('sim', ascending=False)['pattern'].iloc[0]
    return (sim_top)

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

result = spark.createDataFrame(result)
result.show()

# COMMAND ----------

groundtruth = spark.read.format("csv").option("header","true").load("/FileStore/tables/items_pattern.csv").withColumnRenamed('tag','true_tag')
groundtruth.show()

# COMMAND ----------

joined_df = groundtruth.join(result,on='atg_code',how='left')
joined_df = joined_df.withColumn("pattern", coalesce(joined_df['true_tag'], joined_df["tag"])).select('atg_code','pattern')
joined_df = joined_df.withColumn("pattern", when(joined_df["pattern"] == "untagged", "unknown").otherwise(joined_df['pattern']))
joined_df.show()

# COMMAND ----------

truth_with_rule = joined_df.toPandas()
truth_with_rule['pattern'].value_counts()

# COMMAND ----------

index = (truth_with_rule["pattern"] == "unknown")
truth_with_rule.loc[index,'pattern'] = truth_with_rule.loc[index,'atg_code'].apply(lambda x: search_sim(x))
truth_with_rule['pattern'].value_counts()

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

tags = truth_with_rule["pattern"].drop_duplicates().values

for tag in tags:

    df = truth_with_rule[truth_with_rule["pattern"] == tag]

    if len(df) >= 10:

        atgs = np.array(df.sample(10)["atg_code"])

    else:

        atgs = np.array(df["atg_code"])

        print(f"# of items for tag '{tag}': {len(df['atg_code'])}")

    print(atgs)

    show_set(atgs, tag, root)

# COMMAND ----------

images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load('/FileStore/lanecrawford_img')

features_df = images.repartition(16).select(col("path").alias("filePath"),col("content").alias("origin"))


# COMMAND ----------

@udf(returnType=StringType())
def get_code(x):
    return x.split('/')[-1].split('.')[0]

features_df = features_df.withColumn('atg_code',get_code(col('filePath')))
df_with_images = spark.createDataFrame(truth_with_rule)

# COMMAND ----------

df_with_images = df_with_images.join(features_df, on = "atg_code")


# COMMAND ----------

unground_df = df_with_images.join(groundtruth.where(col('tag')=='untagged'), on='atg_code', how='inner' )


# COMMAND ----------

display(unground_df)

# COMMAND ----------

df_with_image.write.csv('/FileStore/tables/df_with_image.csv')

# COMMAND ----------

img = read_image(root, 'BWA941')
plt.imshow(img)

# COMMAND ----------

truth_with_rule = spark.createDataFrame(truth_with_rule)


# COMMAND ----------

display(truth_with_rule.join(groundtruth, on='atg_code', how='left'))

# COMMAND ----------

# MAGIC %md
# MAGIC # 下面的代码都忽略

# COMMAND ----------

for code in truth_with_rule['atg_code'][:1]:
    img = read_image(root, code)
    plt.imshow(img)
    plt.title(truth_with_rule.loc[truth_with_rule['atg_code'] == code,'pattern'].values[0])
    plt.show()
    dbutils.widgets.text("judge", "","Is the tag right (Yes-1, No-0)?")
    judge = int(dbutils.widgets.get("judge"))
    if not judge:
        dbutils.widgets.text("new_tag", "","Input the right tag number: (1-graphic_print; 2-plain; 3-stripe; 4-checks; 5-word_print; 6-multi_color)")
        new_tag = int(dbutils.widgets.get('new_tag'))
        switcher = {1:"graphic_print", 2:"plain", 3: "stripe", 4:"checks", 5:"word_print", 6:"multi_color"}
        truth_with_rule.loc[truth_with_rule['atg_code'] == code, 'pattern'] = switcher[new_tag]
        print(f"{code} is changed into {switcher[new_tag]}")

# COMMAND ----------

pip install pillow

# COMMAND ----------

pip install docx

# COMMAND ----------

pip install docx.shared

# COMMAND ----------

from PIL import Image
from docx import Document
from docx.shared import Inches