# Databricks notebook source
pip install tensorflow

# COMMAND ----------

import pandas as pd
from PIL import Image
import numpy as np
import io

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import *
from tensorflow.keras.applications.nasnet import *
from tensorflow.keras.applications.efficientnet import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

# load a pretrained InceptionV3 model by default with the output layer peeled off 
# model = InceptionV3(include_top=False)
# model = ResNet50(include_top=False)
model = EfficientNetB7(include_top=False)          
model.summary()# verify that the top layer is removed
bc_model_weights = sc.broadcast(model.get_weights())

# COMMAND ----------

def model_fn():
    """
    Returns a InceptionV3 model with top layer removed and broadcasted pretrained weights.
    """
    # model = InceptionV3(weights=None, include_top=False)
    # model = ResNet50(weights=None, include_top=False)
    model = EfficientNetB7(weights=None, include_top=False)
    model.set_weights(bc_model_weights.value)
    return model

def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load('/image_trans_crop')

features_df = images.repartition(16).select(col("path").alias("filePath"),col("content").alias("origin"),featurize_udf("content").alias("features"))

@udf(returnType=StringType())
def get_code(x):
    return x.split('/')[-1].split('_')[0]

features_df = features_df.withColumn('atg_code',get_code(col('filePath')))

# COMMAND ----------

from pyspark.sql.functions import *
features_df.select(size('features')).take(1)

# COMMAND ----------

import os
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
features_df.select('features','atg_code').write.mode("overwrite").parquet(os.path.join(team_path, "feature_df_efficientnet.parquet"))

# COMMAND ----------

labels_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/manual_tag.csv")
labels_df = labels_df.withColumnRenamed('pattern','tag')
#train = labels_df.where(col('true_tag')!='untagged')
#test = labels_df.where(col('true_tag')=='untagged')
train,test = labels_df.randomSplit([0.8,0.2],seed=10)
train.display()
test.display()

# COMMAND ----------

train = train.join(features_df,'atg_code','left').select("features", "tag",'atg_code')
test = test.join(features_df,'atg_code','left').select("features", "tag",'atg_code')
train.display()

# COMMAND ----------

from pyspark.sql.functions import coalesce
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer

label_indexer = StringIndexer(inputCol="tag", outputCol="label")
# train = train.withColumn("label")

model1 = label_indexer.fit(train)
train = model1.transform(train)

model2 = label_indexer.fit(test)
test = model2.transform(test)

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

df_with_vectors_train.head()

# COMMAND ----------

from pyspark.sql.functions import *
features_df.select(size( 'features')) .take(1)

# COMMAND ----------

df_with_vectors_train["features"]

# COMMAND ----------

from pyspark.sql import SparkSession
# solve the problem of Java heap space
spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "45g") \
    .appName('my-cool-app') \
    .getOrCreate()

# COMMAND ----------


from pyspark.ml.feature import PCA
pca = PCA(k=4000,inputCol="features",outputCol="newfeatures")  #Dimensionality reduction from 125440 to 12544
model = pca.fit(df_with_vectors_train)
df_with_vectors_train2=model.transform(df_with_vectors_train)    

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#MLP
layers = [125440,9000, 70, 7]  #input: 125440 features; output: 6 classes
mlp = MultilayerPerceptronClassifier(maxIter=10, layers=layers, featuresCol="features", labelCol="label", predictionCol="prediction", blockSize=128, seed=0)
model = mlp.fit(df_with_vectors_train)

# COMMAND ----------

result = model.transform(df_with_vectors_test)

prediction_label = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print ("MLP test accuracy: " + str(evaluator.evaluate(prediction_label)))

# COMMAND ----------

result.createOrReplaceTempView("result")
spark.sql("select * from result").show(100)

# COMMAND ----------

result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

prediction_label.head()

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Fit LogisticRegression Model
lr = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, regParam=0, elasticNetParam=1, tol=1e-6,threshold=0.5, standardization=True,family="auto")
lrModel = lr.fit(df_with_vectors_train)

# COMMAND ----------




# COMMAND ----------

# Generate predictions on test data 

result = lrModel.transform(df_with_vectors_test)
result.createOrReplaceTempView("result")
spark.sql("select * from result").show(100)
# spark.sql("select filePath, probability, prediction from result").show(100, truncate = False)

# COMMAND ----------

result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

