# Databricks notebook source
import os
import pandas as pd
import numpy as np
# Data path (read-only)

# Raw data
container = "data2"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
#df = spark.read.parquet(os.path.join(data_path, "Ztore-data", "project-data-20230316"))

# Query
df_query = spark.read.parquet(os.path.join(data_path, "Ztore-data", "20230321-training-queries.parquet"))

# Transformed Query
df_query_dummy = spark.read.options(header=True, delimiter=',').csv("dbfs:/queries.csv")
pd_df_query_dummy= df_query_dummy.toPandas()

# COMMAND ----------

pd_df_query_dummy.head()

# COMMAND ----------

def score(cluster):
    import pandas as pd
    # read the original queries
    q = spark.read.options(header=True, delimiter=',').csv("dbfs:/queries.csv")
    q = q.toPandas().astype(int)
    # preprocessing
    df = q.mask(q==0).stack().reset_index(level=1).groupby(level=0).apply(lambda x: sorted(x['level_1'].tolist())).reset_index()
    df.rename(columns={0: 'original'}, inplace=True)
    df['cluster'] = [[v for k, v in cluster.items()] for i in range(len(df))]
    # apply func
    def func(col):
        original = set(col['original'])
        matched_cluster = []
        for cluster in col['cluster']:
            if all(i in original for i in cluster):
                original = original - set(cluster)
                matched_cluster.append(cluster)
        return sorted(original), matched_cluster
    df['split'] = df.apply(func, axis=1)
    result = pd.DataFrame(df['split'].to_list(), columns=['remaining', 'matched_cluster'])
    # calculate result
    original_score = q.sum().sum()
    cluster_score = result['remaining'].apply(len).sum() + result['matched_cluster'].apply(len).sum()
    improved = original_score - cluster_score
    return {'original_score': original_score, 'cluster_score': cluster_score, 'improved': improved}

# COMMAND ----------

# DBTITLE 1,K-mean Billy
# KMeans
# import librabries
from sklearn.cluster import KMeans

def kmeans_algo(pd_df_query_dummy, num_cluster, max_iter, algorithm):
    # data processing
    binary_columns = pd_df_query_dummy.values.transpose() # convert to np array

    col_header = pd_df_query_dummy.columns

    # Cluster the binary columns using k-means with 4 clusters
    kmeans = KMeans(n_clusters=num_cluster, max_iter=max_iter, init='k-means++', algorithm=algorithm, random_state=0).fit(binary_columns)

    # Group the column names based on their k-means cluster assignments
    column_groups = [[] for _ in range(num_cluster)]
    for i, label in enumerate(kmeans.labels_):
        column_name = col_header[i]
        column_groups[label].append(column_name)

    # Combine the column groups into a list of strings
    result = []
    for group in column_groups:
        result.append(" & ".join(group))

    cluster = {n:_ for n, _ in enumerate(column_groups)}
    return cluster


# COMMAND ----------

from itertools import product

# config - combination of hyperparameters
max_iter_list = [100, 200, 300, 400, 500]
n_cluster_min = 15
n_cluster_max = 25
algo_list = ['auto', 'elkan']
distance_list = [False, 'hamming', 'jaccard']
n_cluster_list = [_ for _ in range(n_cluster_min, n_cluster_max + 1)]

# COMMAND ----------

# try using distance as input
from scipy.spatial.distance import cdist, pdist, squareform

# Compute the pairwise Jaccard distance between rows
def generate_jaccard_distance(pd_df_query_dummy):
    binary_columns = pd_df_query_dummy.values.transpose()
    distances = pdist(binary_columns, metric='jaccard')
    distance_matrix = squareform(distances)
    return pd.DataFrame(distance_matrix, columns=pd_df_query_dummy.columns)

def generate_hamming_distance(pd_df_query_dummy):
    binary_columns = pd_df_query_dummy.values.transpose()
    distances = cdist(binary_columns, binary_columns, metric='hamming')
    return pd.DataFrame(distances, columns=pd_df_query_dummy.columns)


# COMMAND ----------

for max_iter, num_cluster, algo, distance in product(*[max_iter_list, n_cluster_list, algo_list, distance_list]):
    if distance_list == 'hamming':
        df_input = generate_hamming_distance(pd_df_query_dummy)
    elif distance_list == 'jaccard':
        df_input = generate_jaccard_distance(pd_df_query_dummy)
    else:
        df_input = pd_df_query_dummy
    cluster = kmeans_algo(df_input, num_cluster, max_iter, algo)
    final_score = score(cluster)
    print(f'distance: {distance}, algo:{algo}, max_iter:{max_iter}, num_cluster:{num_cluster}')
    print(final_score)

# COMMAND ----------

# DBTITLE 1,Hierarchical Clusterin- Tuning Parameter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# import data
binary_columns = pd_df_query_dummy.values.transpose() # convert to np array
col_header = pd_df_query_dummy.columns

# Define list of linkage methods and distance metrics to test
linkage_methods = ['ward', 'complete', 'single', 'average','weighted','centroid']

distance_metrics = ['jaccard', 'hamming', 'dice', 'euclidean', 'cosine']

#distance_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

df_score_hierarchy = pd.DataFrame(columns=['linkage_methods', 'distance_metric', 'threshold','original_score','cluster_score','improved'])

for linkage_method in linkage_methods:
    try:
        for distance_metric in distance_metrics:

                # Compute linkage matrix with Jaccard distance metric
                linkage_matrix = linkage(binary_columns, method=linkage_method, metric=distance_metric)

                for threshold in range(1, 10): #80,101
                    threshold /= 10.0

                    new_row=[]

                    # Assign samples to clusters based on distance threshold
                    #threshold = i
                    hierarchy_cluster = fcluster(linkage_matrix, threshold, criterion='distance')
                    hierarchy_cluster= hierarchy_cluster-1

                    #len(np.unique(clusters))
                    column_groups_hierarchy = [[] for _ in range(len(np.unique(hierarchy_cluster)))]
                    for i, label in enumerate(hierarchy_cluster):
                        column_name = col_header[i]
                        column_groups_hierarchy[label].append(column_name)
                        #print(column_groups_hierarchy)

                    Cluster_hierarachy= {}

                    for n, _ in enumerate(column_groups_hierarchy):
                        Cluster_hierarachy[n]= _
                    
                    new_row.append(linkage_method)
                    new_row.append(distance_metric)
                    new_row.append(threshold)
                    new_row.append(score(Cluster_hierarachy)['original_score'])
                    new_row.append(score(Cluster_hierarachy)['cluster_score'])
                    new_row.append(score(Cluster_hierarachy)['improved'])

                    df_score_hierarchy.loc[len(df_score_hierarchy)] = new_row

                    #name= linkage_method+ "_"+distance_metric+ "_"+str(threshold)
                    #score_hierarachy[name]= score(Cluster_hierarachy)

    except Exception as e:
            print(f"Error occurred: {e}")
            continue

# COMMAND ----------

# DBTITLE 1,Selected the highest improved (747)
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# import data
binary_columns = pd_df_query_dummy.values.transpose() # convert to np array
col_header = pd_df_query_dummy.columns

# Compute linkage matrix with Jaccard distance metric
linkage_matrix = linkage(binary_columns, method='complete', metric='jaccard')

threshold = 0.85

# Assign samples to clusters based on distance threshold
#threshold = i
hierarchy_cluster = fcluster(linkage_matrix, threshold, criterion='distance')
hierarchy_cluster= hierarchy_cluster-1

#len(np.unique(clusters))
column_groups_hierarchy = [[] for _ in range(len(np.unique(hierarchy_cluster)))]
for i, label in enumerate(hierarchy_cluster):
    column_name = col_header[i]
    column_groups_hierarchy[label].append(column_name)
    #print(column_groups_hierarchy)

Cluster_hierarachy= {}

for n, _ in enumerate(column_groups_hierarchy):
    Cluster_hierarachy[n]= _

score_hierarachy= score(Cluster_hierarachy)

# COMMAND ----------

dendrogram(hierarchy_cluster, leaf_rotation=90., leaf_font_size=8.)

# COMMAND ----------

pd_df_query_dummy[pd_df_query_dummy['Gender 2'] == '1']