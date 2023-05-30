# Databricks notebook source
pip install tensorflow

# COMMAND ----------

pip install GPyOpt

# COMMAND ----------

pip install GPy

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
model = ResNet50(include_top=False)
# model = EfficientNetB7(include_top=False)          
model.summary()# verify that the top layer is removed
bc_model_weights = sc.broadcast(model.get_weights())

# COMMAND ----------

# MAGIC %md
# MAGIC # Image Embedding to get Feature Vectors

# COMMAND ----------

def model_fn():
    """
    Returns a InceptionV3 model with top layer removed and broadcasted pretrained weights.
    """
    # model = InceptionV3(weights=None, include_top=False)
    model = ResNet50(weights=None, include_top=False)
    # model = EfficientNetB7(weights=None, include_top=False)
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
    return x.split('/')[-1].split(".")[0].split('_')[0]

features_df = features_df.withColumn('atg_code',get_code(col('filePath')))

# COMMAND ----------

from pyspark.sql.functions import *
features_df.select(size('features')).take(1)

# COMMAND ----------

import os
# Save feature vectors
team_container = "capstone2023-hku-team-b"  # Storage path for teams
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
features_df.select('features','atg_code').write.mode("overwrite").parquet(os.path.join(team_path, "feature_df_resnet.parquet"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Feature Vector from Embedding Step

# COMMAND ----------

import os
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
features_df = spark.read.parquet(os.path.join(team_path, "feature_df_resnet.parquet"))

# COMMAND ----------

# MAGIC %md
# MAGIC Inception V3: "feature_df.parquet" [Row(size(features)=51200)]
# MAGIC
# MAGIC Efficient Net : "feature_df_efficientnet.parquet" [Row(size(features)=125440)]
# MAGIC
# MAGIC ResNet : "feature_df_resnet.parquet" [Row(size(features)=100352)]
# MAGIC
# MAGIC #Load Label from NLP

# COMMAND ----------


labels_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/manual_tag-1.csv")
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
#train,test = labels_df.randomSplit([0.8,0.2],seed=10)
train.display()
test.display()

# COMMAND ----------

labels_df.groupby("tag","label").count().orderBy('count',ascending=False).display()
dictionary = {0:"plain", 1:"graphic_print", 2:"word_print", 3:"stripe", 4: "multi_color",5:"checks"}

# COMMAND ----------

train = train.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")
test = test.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")
# train.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #SMOTE

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

import numpy as np
X_train = np.array(train.select("features").orderBy("atg_code").toPandas()).reshape(-1,1)
Y_train = np.array(train.select("label").orderBy("atg_code").toPandas())

# COMMAND ----------

import pandas as pd
X_new = []
for i in range(len(X_train)):
    X_new.append(X_train[i][0])
X_try = pd.DataFrame(X_new)  

from imblearn.over_sampling import SMOTE,KMeansSMOTE
# sample_strategy = {0: 500, 1: 500, 2: 500, 3: 500, 4: 500, 5: 500}
sample_strategy = "auto"
sm = SMOTE(sampling_strategy=sample_strategy, random_state=10)
X_res, y_res = sm.fit_resample(X_try, Y_train)

X_tr = []
for i in range(len(X_res)):
    X_tr.append([np.array(X_res.iloc[i])])
pd.DataFrame(X_tr)

train_smote = pd.concat([pd.DataFrame(X_tr),pd.DataFrame(y_res)],axis = 1)
train_smote.columns = ['features','label']
train_smote = spark.createDataFrame(train_smote)
smote_df_with_vectors_train = train_smote.select(
    train_smote["label"],
    list_to_vector_udf(train_smote["features"]).alias("features")
)

# COMMAND ----------

smote_df_with_vectors_train.write.mode("overwrite").parquet(os.path.join(team_path, "smote_df_with_vectors_train_resnet.parquet"))

# COMMAND ----------

# MAGIC %md
# MAGIC smote_df_with_vectors_train_inception.parquet
# MAGIC
# MAGIC smote_df_with_vectors_train_efficientnet.parquet
# MAGIC
# MAGIC smote_df_with_vectors_train_resnet.parquet
# MAGIC
# MAGIC
# MAGIC
# MAGIC #Load SMOTE Trainset from Previous Step

# COMMAND ----------

import os
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
smote_df_with_vectors_train = spark.read.parquet(os.path.join(team_path, "smote_df_with_vectors_train_resnet.parquet"))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Grid Search

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder


lr_cv = LogisticRegression(maxIter=30)
paramGrid = ParamGridBuilder()\
				.addGrid(lr_cv.regParam,[0.1,0.05,0.01])\
				.addGrid(lr_cv.elasticNetParam,[0.0,0.1,0.2,0.3,0.4])\
				.build()

# 交叉验证需要输入：待调参的模型、搜索网格、evaluator，交叉验证的折数
crossval = CrossValidator(estimator=lr_cv,
						  estimatorParamMaps=paramGrid,
						  evaluator=MulticlassClassificationEvaluator(),
						  numFolds=5)
lrModel_cv = crossval.fit(smote_df_with_vectors_train)
best_parameters = [(
                [{key.name: paramValue} for key, paramValue in zip(params.keys(), params.values())], metric) \
                for params, metric in zip(
                    lrModel_cv.getEstimatorParamMaps(),
                    lrModel_cv.avgMetrics)]

lr_best_params = sorted(best_parameters, key=lambda el: el[1], reverse=True)[0][0]
print(lr_best_params)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Fit LogisticRegression Model
lr = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, regParam=0.01, elasticNetParam=0.3, tol=1e-6,  standardization=True,family="auto")
lrModel = lr.fit(smote_df_with_vectors_train)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Bayesian Optimization

# COMMAND ----------

import GPy
import GPyOpt
import numpy as np
from GPyOpt.methods import BayesianOptimization
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
bds = [{'name': 'regParam', 'type': 'continuous','domain': [0,0.2]},
       {'name': 'elasticNetParam', 'type': 'continuous', 'domain': [0,0.5]}]
def cv_score(params):
    model = LogisticRegression(maxIter=10)
    reg = params[0][0]
    elastic = params[0][1]
    grid = [{model.regParam: reg, model.elasticNetParam:elastic,model.fitIntercept:True}]
    crossval = CrossValidator(estimator=model,
                          estimatorParamMaps=grid,
						  evaluator=MulticlassClassificationEvaluator(),
						  numFolds=4)
    Model_cv = crossval.fit(smote_df_with_vectors_train)
    metric = Model_cv.avgMetrics
    return metric[0]

def bo_optimization(bds):
    np.random.seed(10)
    kernel = GPy.kern.Matern52(input_dim = 1, variance = 1.0, lengthscale = 1.0)
    optimizer = BayesianOptimization(
        f = cv_score,           # black-box function
        domain = bds,              # parameters domain
        model_type = 'GP',         # GP as surrogate model
        kernel = kernel,           # Matern kernel
        acquisition_type = 'EI',   # expected improvement
        acquisition_jitter = 0.05, # E&E trade-off
        exact_feval = True,      
        maximize = True)
    optimizer.run_optimization(max_iter = 30)
    return optimizer.x_opt

best_para_lr = bo_optimization(bds)
print(best_para_lr)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Fit LogisticRegression Model
lr = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, regParam=best_para_lr[0], elasticNetParam=best_para_lr[1],fitIntercept=True)
lrModel = lr.fit(smote_df_with_vectors_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #Evaluation

# COMMAND ----------

from pyspark.ml.classification import LogisticRegressionModel
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

import mlflow
logged_model = 'dbfs:/databricks/mlflow-tracking/1869162824672013/9a62b036d21c463eb1bfcc7df2615d8d/artifacts/model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# Perform inference via model.transform()
loaded_model.transform(df_with_vectors_test).select("atg_code","prediction","label","tag").orderBy("atg_code").display()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
result = loaded_model.transform(df_with_vectors_test)
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

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
result = lrModel.transform(df_with_vectors_test)
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


y_true = result.select("atg_code","label").orderBy("atg_code")
y_true = y_true.toPandas()
y_pred = result.select("atg_code","prediction").orderBy("atg_code")
y_pred = y_pred.toPandas()


# COMMAND ----------

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix(y_true['label'], y_pred['prediction'])).plot()


# COMMAND ----------

from sklearn.metrics import classification_report
dictionary = {0:"plain", 1:"graphic_print", 2:"word_print", 3:"stripe", 4: "multi_color",5:"checks"}
target_names = ["{}".format(dictionary[i]) for i in range(6)]
print(classification_report(y_true['label'], y_pred['prediction'], 
                            target_names = target_names))

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualization

# COMMAND ----------

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os
def get_image_path(root, atg_code) -> str:

    return os.path.join(root, f"{atg_code}.jpg")

def read_image(root, atg_code):

    image_path = get_image_path(root, atg_code)

    try:

        image = Image.open(image_path).convert("RGB")

        return image

    except Exception as e:

        raise Exception(f"Failed to read image {image_path}") from e

def show_set(set_, tag, root):
    set_atgs = np.array(set_["atg_code"])
    set_label = np.array(set_["label"])
    set_pred = np.array(set_["prediction"])
    fig, axes = plt.subplots(1, len(set_atgs), figsize=(20, 5))

    for i, atg in enumerate(set_atgs):

        img = read_image(root, atg)

        if img:

            axes[i].imshow(img)

            axes[i].set_title(f"{atg}\n pred: {dictionary[int(set_pred[i])]}\n true: {dictionary[int(set_label[i])]}")

            axes[i].grid(False)

            axes[i].axis("off")


    fig.suptitle(f"Prediction: {dictionary[tag]}", fontsize=20)
    
    plt.tight_layout()

    plt.show()

root = "/dbfs/FileStore/lanecrawford_img"
y = result.select("atg_code","prediction","label","tag").orderBy("atg_code").toPandas()
tags = y["prediction"].drop_duplicates().values

for tag in tags:

    df = y[y["prediction"] == tag]

    if len(df) >= 10:

        atgs = (df.sample(10,random_state=10)[["atg_code","prediction","label"]])

    else:

        atgs = (df[["atg_code","prediction","label"]])

        print(f"# of items for tag '{tag}': {len(df['atg_code'])}")

    # print(atgs)
    show_set(atgs, tag, root)
