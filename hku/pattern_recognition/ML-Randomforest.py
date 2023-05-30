# Databricks notebook source
# MAGIC %md
# MAGIC #Inception

# COMMAND ----------

import os
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
#features_df_eff = spark.read.parquet(os.path.join(team_path, "feature_df_efficientnet.parquet"))
#features_df_res = spark.read.parquet(os.path.join(team_path, "feature_df_resnet.parquet"))
features_df = spark.read.parquet(os.path.join(team_path, "feature_df.parquet"))

smote_df_with_vectors_train = spark.read.parquet(os.path.join(team_path, "smote_df_with_vectors_train_inception.parquet"))

# COMMAND ----------

features_df.display()

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt
a = Image.open('/dbfs/image_trans_crop/BVI473_in_xl.jpg')
plt.imshow(a)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/image_trans_crop

# COMMAND ----------

labels_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/manual_tag.csv")
labels_df = labels_df.withColumnRenamed('pattern','tag')

from pyspark.sql.functions import coalesce
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer

label_indexer = StringIndexer(inputCol="tag", outputCol="label")
model1 = label_indexer.fit(labels_df)
labels_df = model1.transform(labels_df)
fractions = labels_df.select("label").distinct().withColumn("fraction", lit(0.8)).rdd.collectAsMap()
train = labels_df.stat.sampleBy("label", fractions, seed=10)
test = labels_df.subtract(train)

# COMMAND ----------

labels_df.groupby("tag","label").count().orderBy('count',ascending=False).display()

# COMMAND ----------

train = train.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")
test = test.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
# since our work-around method using pandas UDF has processed features into series, and LR takes only vector inputs，we can convert array to vector as follows. 
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT()) # The second input item is to specify the output datatype
# Load Test features without label
featuresTestDF = test

# Convert array to vector 

df_with_vectors_test = featuresTestDF.select(
    featuresTestDF['atg_code'],
    featuresTestDF["label"],
    featuresTestDF['tag'],
    list_to_vector_udf(featuresTestDF["features"]).alias("features")
)


# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=200, maxDepth=20)
rfModel = rf.fit(smote_df_with_vectors_train)
result = rfModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

y_true = result.select("atg_code","label").orderBy("atg_code")
y_true = y_true.toPandas()
y_pred = result.select("atg_code","prediction").orderBy("atg_code")
y_pred = y_pred.toPandas()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix(y_true['label'], y_pred['prediction'])).plot()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
result = rfModel.transform(df_with_vectors_test)
evaluator = MulticlassClassificationEvaluator() 

print('F1-Score ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(result,
                   {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(result,   
                   {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'accuracy'}))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #Efficientnet

# COMMAND ----------

import os
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
features_df = spark.read.parquet(os.path.join(team_path, "feature_df_efficientnet.parquet"))
#features_df = spark.read.parquet(os.path.join(team_path, "feature_df_resnet.parquet"))
#features_df = spark.read.parquet(os.path.join(team_path, "feature_df.parquet"))

smote_df_with_vectors_train = spark.read.parquet(os.path.join(team_path, "smote_df_with_vectors_train_efficientnet.parquet"))

# COMMAND ----------

labels_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/manual_tag.csv")
labels_df = labels_df.withColumnRenamed('pattern','tag')

from pyspark.sql.functions import coalesce
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer

label_indexer = StringIndexer(inputCol="tag", outputCol="label")
model1 = label_indexer.fit(labels_df)
labels_df = model1.transform(labels_df)
fractions = labels_df.select("label").distinct().withColumn("fraction", lit(0.8)).rdd.collectAsMap()
train = labels_df.stat.sampleBy("label", fractions, seed=10)
test = labels_df.subtract(train)

# COMMAND ----------

labels_df.groupby("tag","label").count().orderBy('count',ascending=False).display()

# COMMAND ----------

train = train.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")
test = test.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
# since our work-around method using pandas UDF has processed features into series, and LR takes only vector inputs，we can convert array to vector as follows. 
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT()) # The second input item is to specify the output datatype
# Load Test features without label
featuresTestDF = test

# Convert array to vector 
df_with_vectors_test = featuresTestDF.select(
    featuresTestDF['atg_code'],
    featuresTestDF["label"],
    featuresTestDF['tag'],
    list_to_vector_udf(featuresTestDF["features"]).alias("features")
)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=200, maxDepth=20)
rfModel = rf.fit(smote_df_with_vectors_train)
result = rfModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

y_true = result.select("atg_code","label").orderBy("atg_code")
y_true = y_true.toPandas()
y_pred = result.select("atg_code","prediction").orderBy("atg_code")
y_pred = y_pred.toPandas()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix(y_true['label'], y_pred['prediction'])).plot()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
result = rfModel.transform(df_with_vectors_test)
evaluator = MulticlassClassificationEvaluator() 

print('F1-Score ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(result,
                   {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(result,   
                   {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'accuracy'}))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #Resnet

# COMMAND ----------

import os
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
#features_df = spark.read.parquet(os.path.join(team_path, "feature_df_efficientnet.parquet"))
features_df = spark.read.parquet(os.path.join(team_path, "feature_df_resnet.parquet"))
#features_df = spark.read.parquet(os.path.join(team_path, "feature_df.parquet"))

smote_df_with_vectors_train = spark.read.parquet(os.path.join(team_path, "smote_df_with_vectors_train_resnet.parquet"))

# COMMAND ----------

# smote_df_with_vectors_train_inception.parquet

# smote_df_with_vectors_train_efficientnet.parquet

# smote_df_with_vectors_train_resnet.parquet

# COMMAND ----------

labels_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/manual_tag.csv")
labels_df = labels_df.withColumnRenamed('pattern','tag')

from pyspark.sql.functions import coalesce
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer

label_indexer = StringIndexer(inputCol="tag", outputCol="label")
model1 = label_indexer.fit(labels_df)
labels_df = model1.transform(labels_df)
fractions = labels_df.select("label").distinct().withColumn("fraction", lit(0.8)).rdd.collectAsMap()
train = labels_df.stat.sampleBy("label", fractions, seed=10)
test = labels_df.subtract(train)

# COMMAND ----------

labels_df.groupby("tag","label").count().orderBy('count',ascending=False).display()

# COMMAND ----------

train = train.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")
test = test.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
# since our work-around method using pandas UDF has processed features into series, and LR takes only vector inputs，we can convert array to vector as follows. 
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT()) # The second input item is to specify the output datatype
df_with_vectors_train = train.select(
    train['atg_code'],
    train["label"],
    train['tag'],
    list_to_vector_udf(train["features"]).alias("features")
)

# COMMAND ----------

# Load Test features without label
featuresTestDF = test

# Convert array to vector 
df_with_vectors_test = featuresTestDF.select(
    featuresTestDF['atg_code'],
    featuresTestDF["label"],
    featuresTestDF['tag'],
    list_to_vector_udf(featuresTestDF["features"]).alias("features")
)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=200, maxDepth=20)
rfModel = rf.fit(smote_df_with_vectors_train)
result = rfModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

y_true = result.select("atg_code","label").orderBy("atg_code")
y_true = y_true.toPandas()
y_pred = result.select("atg_code","prediction").orderBy("atg_code")
y_pred = y_pred.toPandas()

# COMMAND ----------

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix(y_true['label'], y_pred['prediction'])).plot()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
result = rfModel.transform(df_with_vectors_test)
evaluator = MulticlassClassificationEvaluator() 

print('F1-Score ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(result,
                   {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(result,   
                   {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'accuracy'}))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

