# Databricks notebook source
# DBTITLE 1,Import data
import os
import pandas as pd
import numpy as np
from ast import literal_eval
import json

# Data path (read-only)

# ZStore
container = "data2"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df = spark.read.parquet(os.path.join(data_path, "Ztore-data", "project-data-20230316"))

df_SQL = spark.read.parquet(os.path.join(data_path, "Ztore-data", "20230321-training-queries.parquet"))

df_SQL_Pandas= df_SQL.toPandas()

t_raw = spark.read.parquet(os.path.join(data_path, "Ztore-data", "project-data-20230316"))

# COMMAND ----------

q = spark.read.options(header=True, delimiter=',').csv("dbfs:/queries.csv")
q_pandas= q.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# COMMAND ----------

from sklearn.cluster import KMeans
k = 100  # choose the number of clusters you want
kmeans = KMeans(n_clusters=k, random_state=0).fit(q_pandas)
q_pandas['kmean']= kmeans.labels_

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
# Compute Jaccard similarity coefficient between each pair of observations
jaccard_similarities = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(i+1, 1000):
        jaccard_similarities[i, j] = jaccard_score(X[i], X[j])

# Print average Jaccard similarity coefficient
print(f"Average Jaccard similarity coefficient: {np.mean(jaccard_similarities)}")
wss = []
silhouette_scores = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(q_pandas)
    wss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(q_pandas, kmeans.labels_))

# Plot elbow curve to identify optimal value of k
plt.plot(range(1, 20), wss)
plt.title('Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares')
plt.show()


# Plot silhouette scores to identify optimal value of k
plt.plot(range(2, 20), silhouette_scores)
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

q_pandas= q.toPandas().astype(int)
q_pandas = q_pandas.applymap(lambda x: 1 if x > 0 else 0)
q_pandas = q_pandas.astype(bool)

# Compute the Jaccard similarity matrix
similarity_matrix = 1 - pairwise_distances(q_pandas.values, metric='jaccard')

# Define the range of k values to test
k_range = range(2, 100)

# Fit K-means clustering for each value of k
wcss = []
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(similarity_matrix)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(k_range, wcss)
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares')
plt.title('Elbow method for K-means clustering')
plt.show()

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

q_pandas= q.toPandas().astype(int)
q_pandas = q_pandas.applymap(lambda x: 1 if x > 0 else 0)
q_pandas = q_pandas.astype(bool)

# Compute the Jaccard similarity matrix
similarity_matrix = 1 - pairwise_distances(q_pandas.values, metric='jaccard')

# Define the range of k values to test
k_range = range(60, 200)

# Fit K-means clustering for each value of k
wcss = []
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(similarity_matrix)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(k_range, wcss)
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares')
plt.title('Elbow method for K-means clustering')
plt.show()

# COMMAND ----------

similarity_matrix

# COMMAND ----------

# DBTITLE 1,Transform Json column into list for further operation
from pyspark.sql.functions import col, from_json, explode, collect_list
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType

# Define a schema for the JSON data
json_schema = StructType([
    StructField("list", ArrayType(StructType([
        StructField("item", StringType())
    ])))
])


# Extract the list of items from the struct
df = df.withColumn("active_1_district_list", col("active_1_district.list.item"))
df = df.withColumn("active_2_district_list", col("active_2_district.list.item"))
df = df.withColumn("category_engagement_11_list", col("category_engagement_11.list.item"))
df = df.withColumn("category_engagement_12_list", col("category_engagement_12.list.item"))
df = df.withColumn("category_engagement_13_list", col("category_engagement_13.list.item"))
df = df.withColumn("category_engagement_21_list", col("category_engagement_21.list.item"))
df = df.withColumn("category_engagement_22_list", col("category_engagement_22.list.item"))
df = df.withColumn("category_engagement_31_list", col("category_engagement_31.list.item"))
df = df.withColumn("category_engagement_41_list", col("category_engagement_41.list.item"))
df = df.withColumn("category_engagement_42_list", col("category_engagement_42.list.item"))

# COMMAND ----------

from pyspark.sql.functions import col, when

def age_column(df_original,df_new):
    col_name1 = 'Age Group '
    find_name = 'Age Range '
    for i in range(1, 10):
        col_nm = col_name1 + str(i)
        find = find_name + str(i)
        df_original = df_original.withColumn(col_nm, when(df.age_range == find, "Y").otherwise("N"))
        df_new= df_new.join(df_original.select('user_id', col_nm), on='user_id')
    return df_new



def check_district(col, active_1_district_list, active_2_district_list):
    if col in active_1_district_list or col in active_2_district_list:
        return 'Y'
    else:
        return 'N'



def district_column(df_original, df_new):
    col_name= 'District '

    for i in range(1,11):
        col_nm= col_name + str(i)
        check_district_udf = udf(check_district, StringType())
        df_original = df_original.withColumn(col_nm, check_district_udf(col('col'), col('active_1_district_list'), col('active_2_district_list')))
        df_new= df_new.join(df_original.select('user_id', col_nm), on='user_id')

    return df_new




# COMMAND ----------

from pyspark.sql.functions import udf, col, array_contains
from pyspark.sql.types import StringType

def district_column(df_original, df_new):
    col_name = 'District '

    for i in range(1, 11):
        col = col_name + str(i)
        check_district_udf = udf(lambda active_1, active_2: 'Y' if (array_contains(active_1, col) or array_contains(active_2, col)) else 'N', StringType())
        #df_original = df_original.withColumn(col, check_district_udf(col('active_1_district_list'), col('active_2_district_list')))
        #df_new = df_new.withColumn(col, col)
        print(check_district_udf)
    return df_new

# COMMAND ----------



# COMMAND ----------

df_new = df.select(col("user_id"))
df_new = age_column(df,df_new)
df_new = district_column(df,df_new)

# COMMAND ----------

df_new

# COMMAND ----------

dfq = spark.sql("SELECT * FROM q_raw").toPandas()
dfq