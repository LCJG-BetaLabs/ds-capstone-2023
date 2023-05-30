# Databricks notebook source
import os
import pandas as pd
import numpy as np
import base64
from tensorflow import keras
import seaborn as sns

from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report
from IPython.display import HTML

# COMMAND ----------

container ="data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df_2021 = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2021_BOOKS.csv"), header=True)
df_2022 = spark.read.csv(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_2022_BOOKS.csv"), header=True)
df_2021 = df_2021.toPandas()
df_2022 = df_2022.toPandas()

# Img
df_img_2021 = pd.DataFrame(dbutils.fs.ls("/FileStore/Bookimages"))
df_img_2021["PRODUCT_ID"] = df_img_2021["name"].apply(lambda x: x.replace(".jpg", ""))
df_img_2022 = pd.DataFrame(dbutils.fs.ls("/FileStore/Bookimages2022"))
df_img_2022["PRODUCT_ID"] = df_img_2022["name"].apply(lambda x: x.replace(".jpg", ""))

# COMMAND ----------

# Get only usable img: size != 807
df_img_2021 = df_img_2021[df_img_2021["size"] != 807]
df_img_2022 = df_img_2022[df_img_2022["size"] != 807]

df_img_2021 = df_img_2021[["PRODUCT_ID", "name"]]
df_img_2022 = df_img_2022[["PRODUCT_ID", "name"]]

df_2021 = df_2021[df_2021['PRICE']!='0']
df_2021 = df_2021[~df_2021['QTY_SALES'].isin([None, 'MBI       '])]
df_2022 = df_2022[df_2022['PRICE']!='0']
df_2022 = df_2022[~df_2022['QTY_SALES'].isin([None, 'MBI       '])]

df_2021["QTY_SALES"] = df_2021["QTY_SALES"].astype("int64")
df_2022["QTY_SALES"] = df_2022["QTY_SALES"].astype("int64")
df_2021["STD_SALES"] = minmax_scale(df_2021["QTY_SALES"], feature_range=(0, 1), axis=0, copy=True)
df_2022["STD_SALES"] = minmax_scale(df_2022["QTY_SALES"], feature_range=(0, 1), axis=0, copy=True)
# Remove outlier
# df_2021 = df_2021[df_2021['STD_SALES'] < df_2021["STD_SALES"].quantile(0.85)]

# COMMAND ----------

# Pre-processing
df_2021["PRODUCT_ID"] = df_2021["PRODUCT_ID"].apply(lambda x: x.strip())
df_2022["PRODUCT_ID"] = df_2022["PRODUCT_ID"].apply(lambda x: x.strip())

# COMMAND ----------

df_train = df_2021[["PRODUCT_ID", "QTY_SALES", "STD_SALES"]].merge(df_img_2021, on="PRODUCT_ID")
df_test = df_2022[["PRODUCT_ID", "QTY_SALES", "STD_SALES"]].merge(df_img_2022, on="PRODUCT_ID")

# COMMAND ----------

def categorizeSales(threshold, n):
    return "1" if n >= threshold else "0"

df_train["QTY_SALES"] = df_train["STD_SALES"].apply(lambda x: categorizeSales(0.1, x))
df_test["QTY_SALES"] = df_test["STD_SALES"].apply(lambda x: categorizeSales(0.1, x))

# COMMAND ----------

# ResNet50V2
predict_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

pred_train_generator=predict_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory="/dbfs/FileStore/Bookimages/",
    x_col="name",
    y_col="QTY_SALES",
    batch_size=32,
    class_mode="binary",
    target_size=(224,224))

pred_test_generator=predict_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory="/dbfs/FileStore/Bookimages2022/",
    x_col="name",
    y_col="QTY_SALES",
    batch_size=32,
    class_mode="binary",
    target_size=(224,224))

# COMMAND ----------

base_model = ResNet50V2(weights='imagenet')

# COMMAND ----------

train_pred = base_model.predict(pred_train_generator,pred_train_generator.n//pred_train_generator.batch_size)
test_pred = base_model.predict(pred_test_generator,pred_test_generator.n//pred_test_generator.batch_size)

# COMMAND ----------

df_train_output = df_train.copy(deep=True)
df_train_output["pred"] = keras.applications.resnet50.decode_predictions(
    train_pred, top=1
)
df_train_output = df_train_output[["PRODUCT_ID", "pred"]]

df_test_output = df_test.copy(deep=True)
df_test_output["pred"] = keras.applications.resnet50.decode_predictions(
    test_pred, top=1
)
df_test_output = df_test_output[["PRODUCT_ID", "pred"]]

# COMMAND ----------

train_cls_table_csv = "resnet50v2_train_output.csv"
test_cls_table_csv = "resnet50v2_test_output.csv"

df_train_output.to_csv("/tmp/{}".format(train_cls_table_csv), index=False)
df_test_output.to_csv("/tmp/{}".format(test_cls_table_csv), index=False)

# COMMAND ----------

dbutils.fs.cp("file:/tmp/{}".format(train_cls_table_csv), "dbfs:/FileStore/output/{}".format(train_cls_table_csv))
dbutils.fs.cp("file:/tmp/{}".format(test_cls_table_csv), "dbfs:/FileStore/output/{}".format(test_cls_table_csv))

# COMMAND ----------

def csv_download_button(df, fname):
    csv = df.to_csv(index=False) #if no filename is given, a string is returned
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{fname}">Download CSV File: {fname}</a>'
    display(HTML(href))

# COMMAND ----------

csv_download_button(df_train_output, train_cls_table_csv)
csv_download_button(df_test_output, test_cls_table_csv)

# COMMAND ----------

display(dbutils.fs.ls("/FileStore/output"))

# COMMAND ----------

csv_download_button(df_2021, "df_2021.csv")
csv_download_button(df_2022, "df_2022.csv")

# COMMAND ----------

