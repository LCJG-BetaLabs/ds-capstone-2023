# Databricks notebook source
import os
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"

features_ic_df = spark.read.parquet(os.path.join(team_path, "feature_df.parquet"))
features_en_df = spark.read.parquet(os.path.join(team_path, "feature_df_efficientnet.parquet"))
features_rn_df = spark.read.parquet(os.path.join(team_path, "feature_df_resnet.parquet"))

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier，
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


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

# MAGIC %md
# MAGIC # For Inception V3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create train test df

# COMMAND ----------

import numpy as np
train_ic = train.join(features_ic_df,'atg_code','left').select('atg_code',"features", "tag","label")
test_ic = test.join(features_ic_df,'atg_code','left').select('atg_code',"features", "tag","label")

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
# since our work-around method using pandas UDF has processed features into series, and LR takes only vector inputs，we can convert array to vector as follows. 
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT()) # The second input item is to specify the output datatype
df_with_vectors_train_ic = train_ic.select(
    train_ic['atg_code'],
    train_ic["label"],
    train_ic['tag'],
    list_to_vector_udf(train_ic["features"]).alias("features")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CNN

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Without SMOTE

# COMMAND ----------

layers = [len(df_with_vectors_train_ic.select("features").first()[0]), 128, 128, 7]

# 创建MultilayerPerceptronClassifier分类器，设置层数、标签列、特征列、最大迭代次数和添加softmax层
mlp = MultilayerPerceptronClassifier(layers=layers, labelCol='label', featuresCol='features', maxIter=100, 
                                      blockSize=128, seed=1234, solver="l-bfgs", stepSize=0.03, tol=1e-05, 
                                      initialWeights=None)

# 训练模型
model_mlp = mlp.fit(df_with_vectors_train_ic)


# COMMAND ----------

featuresTestDF_ic = test_ic

# Convert array to vector 
df_with_vectors_test_ic = featuresTestDF_ic.select(
    featuresTestDF_ic['atg_code'],
    featuresTestDF_ic["label"],
    featuresTestDF_ic['tag'],
    list_to_vector_udf(featuresTestDF_ic["features"]).alias("features")
)

# 预测测试集
predictions = model_mlp.transform(df_with_vectors_test_ic)

# 评估模型
# evaluate the model with test set
evaluator = MulticlassClassificationEvaluator() 

print('F1-Score ', evaluator.evaluate(predictions, 
                   {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(predictions,
                   {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(predictions,   
                   {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(predictions, 
                   {evaluator.metricName: 'accuracy'}))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # For EfficientNet

# COMMAND ----------

train_en = train.join(features_en_df,'atg_code','left').select('atg_code',"features", "tag","label")
test_en = test.join(features_en_df,'atg_code','left').select('atg_code',"features", "tag","label")

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
# since our work-around method using pandas UDF has processed features into series, and LR takes only vector inputs，we can convert array to vector as follows. 
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT()) # The second input item is to specify the output datatype
df_with_vectors_train_en = train_en.select(
    train_en['atg_code'],
    train_en["label"],
    train_en['tag'],
    list_to_vector_udf(train_en["features"]).alias("features")
)

# COMMAND ----------

layers = [len(df_with_vectors_train_en.select("features").first()[0]), 128, 128, 7]

# 创建MultilayerPerceptronClassifier分类器，设置层数、标签列、特征列、最大迭代次数和添加softmax层
mlp_en = MultilayerPerceptronClassifier(layers=layers, labelCol='label', featuresCol='features', maxIter=100, 
                                      blockSize=128, seed=1234, solver="l-bfgs", stepSize=0.03, tol=1e-05, 
                                      initialWeights=None)

# 训练模型
model_mlp_en = mlp_en.fit(df_with_vectors_train_en)


# COMMAND ----------

featuresTestDF_en = test_en

# Convert array to vector 
df_with_vectors_test_en = featuresTestDF_en.select(
    featuresTestDF_en['atg_code'],
    featuresTestDF_en["label"],
    featuresTestDF_en['tag'],
    list_to_vector_udf(featuresTestDF_en["features"]).alias("features")
)

# 预测测试集
predictions = model_mlp_en.transform(df_with_vectors_test_en)

# 评估模型
# evaluate the model with test set
evaluator = MulticlassClassificationEvaluator() 

print('F1-Score ', evaluator.evaluate(predictions, 
                   {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(predictions,
                   {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(predictions,   
                   {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(predictions, 
                   {evaluator.metricName: 'accuracy'}))
                   

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # For ResNet

# COMMAND ----------

train_rn = train.join(features_rn_df,'atg_code','left').select('atg_code',"features", "tag","label")
test_rn = test.join(features_rn_df,'atg_code','left').select('atg_code',"features", "tag","label")

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
# since our work-around method using pandas UDF has processed features into series, and LR takes only vector inputs，we can convert array to vector as follows. 
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT()) # The second input item is to specify the output datatype
df_with_vectors_train_rn = train_rn.select(
    train_rn['atg_code'],
    train_rn["label"],
    train_rn['tag'],
    list_to_vector_udf(train_rn["features"]).alias("features")
)

# COMMAND ----------

layers = [len(df_with_vectors_train_rn.select("features").first()[0]), 128, 128, 7]

# 创建MultilayerPerceptronClassifier分类器，设置层数、标签列、特征列、最大迭代次数和添加softmax层
mlp_rn = MultilayerPerceptronClassifier(layers=layers, labelCol='label', featuresCol='features', maxIter=100, 
                                      blockSize=128, seed=1234, solver="l-bfgs", stepSize=0.03, tol=1e-05, 
                                      initialWeights=None)

# 训练模型
model_mlp_rn = mlp_rn.fit(df_with_vectors_train_rn)

# COMMAND ----------

featuresTestDF_rn = test_rn

# Convert array to vector 
df_with_vectors_test_rn = featuresTestDF_rn.select(
    featuresTestDF_rn['atg_code'],
    featuresTestDF_rn["label"],
    featuresTestDF_rn['tag'],
    list_to_vector_udf(featuresTestDF_rn["features"]).alias("features")
)

# 预测测试集
predictions = model_mlp_rn.transform(df_with_vectors_test_rn)

# 评估模型
# evaluate the model with test set
evaluator = MulticlassClassificationEvaluator() 

print('F1-Score ', evaluator.evaluate(predictions, 
                   {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(predictions,
                   {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(predictions,   
                   {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(predictions, 
                   {evaluator.metricName: 'accuracy'}))