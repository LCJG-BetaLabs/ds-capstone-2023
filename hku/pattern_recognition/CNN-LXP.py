# Databricks notebook source
import os
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"

features_ic_df = spark.read.parquet(os.path.join(team_path, "feature_df.parquet"))
features_en_df = spark.read.parquet(os.path.join(team_path, "feature_df_efficientnet.parquet"))
features_rn_df = spark.read.parquet(os.path.join(team_path, "feature_df_resnet.parquet"))

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#from pyspark.ml.layers import Convolution, MaxPooling, Flatten, Dense, Dropout

# COMMAND ----------

labels_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/manual_tag.csv")
labels_df = labels_df.withColumnRenamed('pattern','tag')

from pyspark.sql.functions import coalesce
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer

label_indexer = StringIndexer(inputCol="tag", outputCol="label")
model1 = label_indexer.fit(labels_df)
labels_df = model1.transform(labels_df)
#train = labels_df.where(col('true_tag')!='untagged')
#test = labels_df.where(col('true_tag')=='untagged')
train,test = labels_df.randomSplit([0.8,0.2],seed=10)


# COMMAND ----------

# MAGIC %md
# MAGIC # For Inception V3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create train test df

# COMMAND ----------

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

len(df_with_vectors_train_ic.select("features").first()[0])

# COMMAND ----------

from pyspark.sql import SparkSession
# solve the problem of Java heap space
spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "45g") \
    .appName('my-cool-app') \
    .getOrCreate()

# COMMAND ----------

layers = [len(df_with_vectors_train_ic.select("features").first()[0]), 252, 126, 7]

# 创建MultilayerPerceptronClassifier分类器，设置层数、标签列、特征列、最大迭代次数和添加softmax层
mlp = MultilayerPerceptronClassifier(layers=layers, labelCol='label', featuresCol='features', maxIter=50, 
                                      blockSize=126, seed=42, solver="l-bfgs", stepSize=0.05, tol=1e-05, 
                                      initialWeights=None)

# 训练模型
model_mlp = mlp.fit(df_with_vectors_train_ic)


#[len(df_with_vectors_train_ic.select("features").first()[0]), 256, 128, 7]：Accuracy  0.6858638743455497
#[len(df_with_vectors_train_ic.select("features").first()[0]), 256, 128,56, 7]: Accuracy  0.6335078534031413
#[len(df_with_vectors_train_ic.select("features").first()[0]), 256, 256, 7]Accuracy  0.6858638743455497
#[len(df_with_vectors_train_ic.select("features").first()[0]), 300, 256, 7]Accuracy  0.675392670157068
# [len(df_with_vectors_train_ic.select("features").first()[0]), 252, 126, 7] blockSize=126 Accuracy  0.6963350785340314
#[len(df_with_vectors_train_ic.select("features").first()[0]), 252, 126, 7]blockSize=126  stepSize=0.05 Accuracy  0.7120418848167539
# [len(df_with_vectors_train_ic.select("features").first()[0]), 252, 126, 7]blockSize=126  stepSize=0.06 seed=42 0.6963350785340314

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

