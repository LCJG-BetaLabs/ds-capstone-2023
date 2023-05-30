# Databricks notebook source
# MAGIC %run ./func

# COMMAND ----------

z = Zstore()

# COMMAND ----------

affinity = []
for i in ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors', 'jaccard']:
    t = z.spectral_clustering(n_clusters=30, affinity=i, assign_labels='kmeans', nlargest=2)
    affinity.append((i, t['improved_score']))
    display((i, t['improved_score']))