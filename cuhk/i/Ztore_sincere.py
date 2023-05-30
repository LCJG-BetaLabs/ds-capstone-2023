# Databricks notebook source
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

q = spark.read.options(header=True, delimiter=',').csv("dbfs:/queries.csv")
ztore_data = q.toPandas()

# COMMAND ----------

from sklearn.cluster import Birch
num_cluster = 95

brc = Birch(n_clusters=num_cluster)
brc_ztore = brc.fit(ztore_data.transpose())

col_header = ztore_data.columns
column_groups = [[] for _ in range(num_cluster)]
for i, label in enumerate(brc_ztore.labels_):
    column_name = col_header[i]
    column_groups[label].append(column_name)

# Combine the column groups into a list of strings
result = []
for group in column_groups:
    result.append(" & ".join(group))
cluster = {n:_ for n, _ in enumerate(column_groups)}


# COMMAND ----------

cluster

# COMMAND ----------

pca = PCA(n_components=6)
comp = pca.fit(ztore_data).components_
comp = pd.DataFrame(comp).transpose()
comp.columns = ['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6']
comp.index = ztore_data.columns

var_ratio = pca.explained_variance_ratio_
var_ratio= pd.DataFrame(var_ratio).transpose()
var_ratio.columns = ['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6']
var_ratio.index = ['Proportion of Variance']
print(var_ratio)

pcomp = pca.fit_transform(ztore_data)
pcomp = pd.DataFrame(pcomp)
pcomp = pcomp.iloc[:,0:6]
pcomp.columns = ['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6']
sns.pairplot(pcomp)


# COMMAND ----------


pca = PCA(n_components='mle')
pca_ztore = pca.fit(ztore_data)
ztore_transform = pca.transform(ztore_data)

# COMMAND ----------

print('Original Shape:', ztore_data.shape)
print('Transform Shape:', ztore_transform.shape)
print(pca_ztore.singular_values_)

ztore_transform
