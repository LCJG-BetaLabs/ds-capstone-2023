# Databricks notebook source
# MAGIC %run ./func

# COMMAND ----------

z = Zstore()

# COMMAND ----------

z.dbscan(eps=0.2, min_samples=5, metric='correlation', algorithm='auto', nlargest=2)

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

