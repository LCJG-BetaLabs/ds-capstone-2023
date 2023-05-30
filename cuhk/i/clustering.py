# Databricks notebook source
# MAGIC %md
# MAGIC # This script is to call the function to fit the data into different clustering algorithms and tune the parameters

# COMMAND ----------

# MAGIC %run ./func

# COMMAND ----------

z = Zstore()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Importance

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
q = spark.read.options(header=True, delimiter=',').csv("dbfs:/queries.csv").toPandas().astype(int)
q['cost'] = q.sum(axis=1)
X, y = q.iloc[:,:-1], q['cost']
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)
feat_importances = pd.Series(regr.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

# COMMAND ----------

# MAGIC %md
# MAGIC # Agglomerative Clustering

# COMMAND ----------

#Baseline model
z.agglomerative_clustering(n_clusters=31, affinity='cosine', linkage='complete', nlargest=2)

# COMMAND ----------

affinity = []
for i in ['euclidean', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'cosine']:
    t = z.agglomerative_clustering(n_clusters=30, affinity=i, linkage='complete', nlargest=2)
    affinity.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

linkage = []
for i in ['complete', 'average', 'single']:
    t = z.agglomerative_clustering(n_clusters=30, affinity='correlation', linkage=i, nlargest=2)
    linkage.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

nlargest = []
for i in range(2, 11):
    t = z.agglomerative_clustering(n_clusters=30, affinity='correlation', linkage='complete', nlargest=i)
    nlargest.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

n_clusters = []
for i in range(1, 101):
    t = z.agglomerative_clustering(n_clusters=i, affinity='correlation', linkage='complete', nlargest=2)
    n_clusters.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

# MAGIC %md
# MAGIC # DBSCAN

# COMMAND ----------

#Baseline
z.dbscan(eps=0.2, min_samples=5, metric='mahalanobis', algorithm='auto', nlargest=2)

# COMMAND ----------

metric = []
for i in ['euclidean', 'canberra', 'chebyshev', 'correlation', 'mahalanobis']:
    t = z.dbscan(eps=0.2, min_samples=5, metric=i, algorithm='auto', nlargest=2)
    metric.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

eps = []
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    t = z.dbscan(eps=i, min_samples=5, metric='correlation', algorithm='auto', nlargest=2)
    eps.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

z.dbscan(eps=0.2, min_samples=5, metric='correlation', algorithm='auto', nlargest=2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Spectral Clustering

# COMMAND ----------

#Baseline
z.spectral_clustering(n_clusters=30, affinity='nearest_neighbors', assign_labels='kmeans', nlargest=2)

# COMMAND ----------

affinity = []
for i in ['nearest_neighbors']:
    t = z.spectral_clustering(n_clusters=30, affinity=i, assign_labels='kmeans', nlargest=2)
    affinity.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

assign_labels = []
for i in ['kmeans', 'discretize']:
    t = z.spectral_clustering(n_clusters=30, affinity='nearest_neighbors', assign_labels=i, nlargest=2)
    assign_labels.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

n_clusters = []
for i in range(1, 101):
    t = z.spectral_clustering(n_clusters=i, affinity='nearest_neighbors', assign_labels='discretize', nlargest=2)
    n_clusters.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

metric = []
for i in ['euclidean', 'canberra', 'chebyshev', 'correlation', 'mahalanobis']:
    t = z.dbscan(eps=0.2, min_samples=5, metric=i, algorithm='auto', nlargest=2)
    metric.append((i, t['improved_score']))
    display((i, t['improved_score']))

# COMMAND ----------

n_clusters = []
for i in range(1, 101):
    t = z.spectral_clustering(n_clusters=i, affinity='nearest_neighbors', assign_labels='kmeans', nlargest=2)
    n_clusters.append((i, t['improved_score']))
    display((i, t['improved_score']))