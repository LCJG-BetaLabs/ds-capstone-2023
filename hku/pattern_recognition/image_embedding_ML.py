# Databricks notebook source
import os
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
#features_df_eff = spark.read.parquet(os.path.join(team_path, "feature_df_efficientnet.parquet"))
features_df = spark.read.parquet(os.path.join(team_path, "feature_df_resnet.parquet"))
smote_df_with_vectors_train = spark.read.parquet(os.path.join(team_path, "smote_df_with_vectors_train_resnet.parquet"))

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
rfModel = rf.fit(df_with_vectors_train)
result = rfModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

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

dictionary = {0:"plain", 1:"graphic_print", 2:"word_print", 3:"stripe", 4: "multi_color",5:"checks"}

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

    print(atgs)
    show_set(atgs, tag, root)


# COMMAND ----------

-------

# COMMAND ----------

pip install sparkxgb

# COMMAND ----------

import os
# Storage path for teams
team_container = "capstone2023-hku-team-b"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
#features_df = spark.read.parquet(os.path.join(team_path, "feature_df_efficientnet.parquet"))
features_df = spark.read.parquet(os.path.join(team_path, "feature_df_resnet.parquet"))
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
#train = labels_df.where(col('true_tag')!='untagged')
#test = labels_df.where(col('true_tag')=='untagged')
train,test = labels_df.randomSplit([0.8,0.2],seed=10)
train.display()
test.display()

# COMMAND ----------

labels_df.groupby("tag","label").count().orderBy('count',ascending=False).display()

# COMMAND ----------

train = train.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")
test = test.join(features_df,'atg_code','left').select('atg_code',"features", "tag","label")
train.display()

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
# sample_strategy = {0: 400, 1: 400, 2: 400, 3: 400, 4: 400, 5: 400}
sample_strategy = "auto"
sm = SMOTE(sampling_strategy=sample_strategy, random_state=10)
X_res, y_res = sm.fit_resample(X_try, Y_train)

X_tr = []
for i in range(len(X_res)):
    X_tr.append([np.array(X_res.iloc[i])])
pd.DataFrame(X_tr)

train_smote = pd.concat([pd.DataFrame(X_tr),pd.DataFrame(y_res)],axis = 1)
train_smote.columns = ['features','label']
print(train_smote.shape)
train_smote = spark.createDataFrame(train_smote)
smote_df_with_vectors_train = train_smote.select(
    train_smote["label"],
    list_to_vector_udf(train_smote["features"]).alias("features")
)

smote_df_with_vectors_train = spark.read.parquet(os.path.join(team_path, "smote_df_with_vectors_train_resnet.parquet"))

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

from pyspark.ml.classification import LogisticRegression

# Fit LogisticRegression Model
lr = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, regParam=0.01, elasticNetParam=1, tol=1e-6,threshold=0.5, standardization=True,family="auto")
lrModel = lr.fit(df_with_vectors_train)

# Generate predictions on test data 

result = lrModel.transform(df_with_vectors_test)
result.createOrReplaceTempView("result")
spark.sql("select * from result").show(100)
# spark.sql("select filePath, probability, prediction from result").show(100, truncate = False)


# COMMAND ----------

result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(maxDepth=20) 
dtModel = dt.fit(df_with_vectors_train)
result = dtModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=200, maxDepth=20)
rfModel = rf.fit(df_with_vectors_train)
result = rfModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import LinearSVC
svm = LinearSVC()
ovr = OneVsRest(classifier=svm)
model = ovr.fit(df_with_vectors_train)
result = model.transform(df_with_vectors_test)
# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from sparkxgb import XGBoostClassifier
from pyspark.ml import Pipeline
xgboost = XGBoostClassifier(
    eta= 0.05,
    gamma= 0.1,
    numRound=50,
    colsampleBytree= 0.8,
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        objective='multi:softprob',
        numClass=6,
        missing=0.0
    )


pipeline = Pipeline(stages=[xgboost])
xgb_model = pipeline.fit(smote_df_with_vectors_train)
result = xgb_model.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

---

# COMMAND ----------

from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import LinearSVC
svm = LinearSVC()
svm.setMaxIter(50)
ovr = OneVsRest(classifier=svm)
model = ovr.fit(df_with_vectors_train)
result = model.transform(df_with_vectors_test)
# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(maxDepth=20) 
dtModel = dt.fit(df_with_vectors_train)
result = dtModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=100, maxDepth=20)
rfModel = rf.fit(df_with_vectors_train)
result = rfModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from sparkxgb import XGBoostClassifier
from pyspark.ml import Pipeline
xgboost = XGBoostClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        objective='multi:softprob',
        numClass=6,
        missing=0.0
    )
pipeline = Pipeline(stages=[xgboost])
xgb_model = pipeline.fit(df_with_vectors_train)
result = xgb_model.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

##以下为smote版本

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Fit LogisticRegression Model
lr = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, regParam=0.01, elasticNetParam=1, tol=1e-6,threshold=0.5, standardization=True,family="auto")
lrModel = lr.fit(smote_df_with_vectors_train)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegressionModel

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

# Generate predictions on test data 

result = lrModel.transform(df_with_vectors_test)
result.createOrReplaceTempView("result")
spark.sql("select * from result").show(100)
# spark.sql("select filePath, probability, prediction from result").show(100, truncate = False)

# COMMAND ----------

result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# evaluate the model with test set
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
target_names = ["Class {}".format(i) for i in range(7)]
print(classification_report(y_true['label'], y_pred['prediction'], 
                            target_names = target_names))

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(maxDepth=20) 
dtModel = dt.fit(smote_df_with_vectors_train)
result = dtModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=100, maxDepth=20)
rfModel = rf.fit(smote_df_with_vectors_train)
result = rfModel.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

pip install sparkxgb

# COMMAND ----------

from sparkxgb import XGBoostClassifier
from pyspark.ml import Pipeline
xgboost = XGBoostClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        objective='multi:softprob',
        numClass=6,
        missing=0.0
    )
pipeline = Pipeline(stages=[xgboost])
xgb_model = pipeline.fit(smote_df_with_vectors_train)
result = xgb_model.transform(df_with_vectors_test)

# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import LinearSVC
svm = LinearSVC()
ovr = OneVsRest(classifier=svm)
model = ovr.fit(smote_df_with_vectors_train)
result = model.transform(df_with_vectors_test)
# accuracy
result.filter(result.label == result.prediction).count()/result.count()

# COMMAND ----------

----------

# COMMAND ----------

result = model.transform(df_with_vectors_test)

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

