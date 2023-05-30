# Databricks notebook source
# MAGIC %md
# MAGIC # This script is to create a set of functions for fitting cluestering algorithm and calculating the improvement

# COMMAND ----------

from IPython.display import display
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
pd.options.display.max_colwidth = 10000

# COMMAND ----------

class Zstore:

    def __init__(self):
        # read the original queries
        self.q = spark.read.options(header=True, delimiter=',').csv("dbfs:/queries.csv").toPandas().astype(int)
        # Drop the query column and convert to numpy array
        self.X = self.q.to_numpy()
        # Standardize the data
        self.X_std = StandardScaler().fit_transform(self.X)

    def get_score(self, clusters):
        # preprocessing
        df = self.q.mask(self.q==0).stack()
        df = df.reset_index(level=1)
        df = df.groupby(level=0).apply(lambda x: sorted(x['level_1'].tolist())).reset_index()
        df.rename(columns={0: 'original'}, inplace=True)
        df['cluster'] = [[v for k, v in clusters.items()] for i in range(len(df))]
        # apply func
        def func(col):
            original = set(col['original'])
            matched_cluster = []
            for cluster in col['cluster']:
                if all(i in original for i in cluster):
                    matched_cluster.append(cluster)
            if matched_cluster: 
                remaining = set.intersection(*[original - set(matched) for matched in matched_cluster])
            else:
                remaining = original
            return sorted(original), sorted(remaining), matched_cluster
        df['split'] = df.apply(func, axis=1)
        result = pd.DataFrame(df['split'].to_list(), columns=['original', 'remaining', 'matched'])
        # calculate result
        result['original_score'] = result['original'].apply(len)
        result['remaining_score'] = result['remaining'].apply(len)
        result['matched_score'] = result['matched'].apply(len)
        original_score = result['original_score'].sum()
        cluster_score = result['remaining_score'].sum() + result['matched_score'].sum()
        return {'original_score': original_score, 'cluster_score': cluster_score, 'improved_score': original_score - cluster_score, 'df': result}

    def get_score_by_clusters(self, clusters):
        result = {}
        for _, cluster in clusters.items():
            result[' | '.join(cluster)] = self.get_score({'': cluster})['improved_score']
        return result

    def get_cluster_result(self, clustering, nlargest):
        queries = self.q.copy()
        # Get the cluster labels and add them to the original dataset
        labels = clustering.labels_
        queries['cluster'] = labels
        # Group the queries by cluster and count the occurrences of each column
        grouped = queries.groupby('cluster').sum()
        # Get the columns used most frequently in each cluster
        clusters = {}
        for i in range(len(grouped)):
            top_columns = grouped.iloc[i].nlargest(nlargest)
            clusters[f'Cluster {i+1}'] = sorted(top_columns.index.tolist())
        # Output
        output = {}
        output['cluster'] = {k: [i for _, i in v.items()] for k, v in pd.DataFrame(clusters).T.drop_duplicates(keep='first').T.to_dict().items()}
        output['no_of_cluster'] = len(output['cluster'])
        output['improved_score'] = self.get_score(clusters)["improved_score"]
        return output

    def agglomerative_clustering(self, n_clusters=29, affinity='jaccard', linkage='complete', nlargest=2):
        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        clustering.fit(self.X_std)
        # Get cluster result
        return self.get_cluster_result(clustering, nlargest)

    def kmeans(self, n_clusters=30, nlargest=2):
        # Perform clustering
        clustering = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
        clustering.fit(self.X_std)
        # Get cluster result
        return self.get_cluster_result(clustering, nlargest)
    
    def spectral_clustering(self, n_clusters=30, affinity='nearest_neighbors', assign_labels='kmeans', nlargest=2):
        # Perform clustering
        clustering = SpectralClustering(n_clusters=n_clusters, affinity=affinity, assign_labels=assign_labels, random_state=0)
        clustering.fit(self.X_std)
        # Get cluster result
        return self.get_cluster_result(clustering, nlargest)

    def dbscan(self, eps=3, min_samples=2, metric='jaccard', algorithm='auto', nlargest=2):
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm)
        clustering.fit(self.X_std)
        # Get cluster result
        return self.get_cluster_result(clustering, nlargest)