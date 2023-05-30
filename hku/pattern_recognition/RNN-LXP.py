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
# MAGIC ## RNN

# COMMAND ----------

len(df_with_vectors_train_ic.select("features").first()[0])

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col

# 转换数据形状为适合RNN的形状（样本数，时间步数，特征数）
time_step = 10
input_col = "features"
output_col = "label"
df_with_vectors_train_ic_rdd = df_with_vectors_train_ic.rdd.map(lambda x: (Vectors.dense(x[input_col]), x[output_col]))
df_with_vectors_train_ic_rdd_time = df_with_vectors_train_ic_rdd.flatMap(lambda x: [(x[0], x[1])] + \
                                             [(Vectors.dense([0.0] * len(x[0])), None) for i in range(time_step-1)])
df_with_vectors_train_ic_rnn = df_with_vectors_train_ic_rdd_time.toDF([input_col, output_col])
assembler = VectorAssembler(inputCols=[input_col], outputCol="features_assembled")
df_with_vectors_train_ic_rnn = assembler.transform(df_with_vectors_train_ic_rnn)\
                                     .select(col("features_assembled"), col(output_col))\
                                     .withColumnRenamed("features_assembled", "features")



# 创建RNN模型
from pyspark.ml.classification import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, TimeDistributed,Bidirectional
import numpy as np
import pandas as pd

input_col = "features"
output_col = "label"
rnn_layers = [128, 128, 7]
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=rnn_layers[0], return_sequences=True, input_shape=(time_step, len(df_with_vectors_train_ic.select("features").first()[0]))))
for i in range(1, len(rnn_layers)):
    rnn_model.add(SimpleRNN(units=rnn_layers[i], return_sequences=True))
rnn_model.add(TimeDistributed(Dense(7, activation='softmax')))
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# COMMAND ----------

# 训练模型
df_with_vectors_train_ic_rnn_list = df_with_vectors_train_ic_rnn.collect()
df_with_vectors_train_ic_rnn_pd = pd.DataFrame(df_with_vectors_train_ic_rnn_list, columns=df_with_vectors_train_ic_rnn.columns)
X = np.array(df_with_vectors_train_ic_rnn_pd[input_col].tolist())
Y = np.array(df_with_vectors_train_ic_rnn_pd[output_col].tolist())
Y.shape
rnn_model.fit(X, Y, epochs=10, batch_size=128)

# COMMAND ----------

X.shape

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Conv2D,MaxPooling2D,Activation,Dropout
datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                    width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")

# COMMAND ----------

model = Sequential()
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=7, activation='softmax'))

# COMMAND ----------

#model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# COMMAND ----------

his=model.fit(datagen.flow(X,Y,batch_size=32),epochs=10)

# COMMAND ----------

from keras.utils import to_categorical
Y = to_categorical(Y)
Y.shape

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
predictions = rnn_model.fit(X, Y, epochs=10, batch_size=128).transform(df_with_vectors_test_ic)

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

