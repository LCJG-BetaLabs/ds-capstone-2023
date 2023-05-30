# Databricks notebook source
pip install sparkxgb

# COMMAND ----------

import os
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
features_en_df = spark.read.parquet(os.path.join(team_path, "feature_df_efficientnet.parquet"))
features_rn_df = spark.read.parquet(os.path.join(team_path, "feature_df_resnet.parquet"))
features_ic_df = spark.read.parquet(os.path.join(team_path, "feature_df.parquet"))

smote_df_with_vectors_train = spark.read.parquet(os.path.join(team_path, "smote_df_with_vectors_train_resnet.parquet"))

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

# MAGIC %md
# MAGIC # For Resnet

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
train_rn = train.join(features_rn_df,'atg_code','left').select('atg_code',"features", "tag","label")
test_rn = test.join(features_rn_df,'atg_code','left').select('atg_code',"features", "tag","label")
train_rn = train_rn.withColumn("label", col("label").cast(IntegerType()))
test_rn = test_rn.withColumn("label", col("label").cast(IntegerType()))

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
 
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT()) # The second input item is to specify the output datatype
df_with_vectors_train = train_rn.select(
    # train_ic['atg_code'],
    train_rn["label"],
    # train_ic['tag'],
    list_to_vector_udf(train_rn["features"]).alias("features")
)

from sparkxgb import XGBoostClassifier
from pyspark.ml import Pipeline
xgboost = XGBoostClassifier(
    eta= 0.05,
    gamma= 0.1,
    numRound=200,
    colsampleBytree= 0.8,
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        objective='multi:softprob',
        numClass=6,
        missing=0.0
    )

# Load Test features without label
featuresTestDF = test_rn

# Convert array to vector 
df_with_vectors_test = featuresTestDF.select(
    featuresTestDF['atg_code'],
    featuresTestDF["label"],
    featuresTestDF['tag'],
    list_to_vector_udf(featuresTestDF["features"]).alias("features")
)

pipeline = Pipeline(stages=[xgboost])
xgb_model = pipeline.fit(df_with_vectors_train)
result = xgb_model.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

#### Result: 0.680628272251309

# COMMAND ----------

# accuracy
evaluator = MulticlassClassificationEvaluator() 

print('F1-Score ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(result,
                   {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(result,   
                   {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'accuracy'}))

# Result
#### F1-Score  0.6145552904092471
#### Precision  0.6250351644916777
#### Recall  0.6806282722513088
#### Accuracy  0.680628272251309

# COMMAND ----------

# MAGIC %md
# MAGIC # For Inception v3

# COMMAND ----------

train_ic = train.join(features_ic_df,'atg_code','left').select('atg_code',"features", "tag","label")
test_ic = test.join(features_ic_df,'atg_code','left').select('atg_code',"features", "tag","label")
train_ic = train_ic.withColumn("label", col("label").cast(IntegerType()))
test_ic = test_ic.withColumn("label", col("label").cast(IntegerType()))

# COMMAND ----------

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT()) # The second input item is to specify the output datatype
df_with_vectors_train = train_ic.select(
    # train_ic['atg_code'],
    train_ic["label"],
    # train_ic['tag'],
    list_to_vector_udf(train_ic["features"]).alias("features")
)


xgboost = XGBoostClassifier(
    eta= 0.05,
    gamma= 0.1,
    numRound=200,
    colsampleBytree= 0.8,
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        objective='multi:softprob',
        numClass=6,
        missing=0.0
    )

# Load Test features without label
featuresTestDF = test_ic

# Convert array to vector 
df_with_vectors_test = featuresTestDF.select(
    featuresTestDF['atg_code'],
    featuresTestDF["label"],
    featuresTestDF['tag'],
    list_to_vector_udf(featuresTestDF["features"]).alias("features")
)

pipeline = Pipeline(stages=[xgboost])
xgb_model = pipeline.fit(df_with_vectors_train)
result = xgb_model.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

#### Result: 0.6335078534031413

# COMMAND ----------

# accuracy
evaluator = MulticlassClassificationEvaluator() 

print('F1-Score ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(result,
                   {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(result,   
                   {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'accuracy'}))

# Result
#### F1-Score  0.5678007609589749
#### Precision  0.5838444278234853
#### Recall  0.6335078534031413
#### Accuracy  0.6335078534031413

# COMMAND ----------

# MAGIC %md
# MAGIC # For Efficientnet

# COMMAND ----------

train_en = train.join(features_en_df,'atg_code','left').select('atg_code',"features", "tag","label")
test_en = test.join(features_en_df,'atg_code','left').select('atg_code',"features", "tag","label")
train_en = train_en.withColumn("label", col("label").cast(IntegerType()))
test_en = test_en.withColumn("label", col("label").cast(IntegerType()))

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from sparkxgb import XGBoostClassifier
from pyspark.ml import Pipeline

# COMMAND ----------

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT()) # The second input item is to specify the output datatype
df_with_vectors_train = train_en.select(
    
    train_en["label"],

    list_to_vector_udf(train_en["features"]).alias("features")
)


xgboost = XGBoostClassifier(
    eta= 0.05,
    gamma= 0.1,
    numRound=200,
    colsampleBytree= 0.8,
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        objective='multi:softprob',
        numClass=6,
        missing=0.0
    )

# Load Test features without label
featuresTestDF = test_en

# Convert array to vector 
df_with_vectors_test = featuresTestDF.select(
    featuresTestDF['atg_code'],
    featuresTestDF["label"],
    featuresTestDF['tag'],
    list_to_vector_udf(featuresTestDF["features"]).alias("features")
)

pipeline = Pipeline(stages=[xgboost])
xgb_model = pipeline.fit(df_with_vectors_train)
result = xgb_model.transform(df_with_vectors_test)

# Result: 0.643979057591623

# COMMAND ----------

# accuracy
evaluator = MulticlassClassificationEvaluator() 

print('F1-Score ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(result,
                   {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(result,   
                   {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(result, 
                   {evaluator.metricName: 'accuracy'}))

# Result
#### F1-Score  0.5712879243810871
#### Precision  0.6053440160770003
#### Recall  0.643979057591623
#### Accuracy  0.643979057591623