# Databricks notebook source
# MAGIC %md # this notebook shows the steps to train a binary ANN keras classifier that:
# MAGIC   - takes in `description` of a book as predictor
# MAGIC   - predicts a binary variable `is_high_sales_volume` (flagged as 1 or 0)
# MAGIC     - `is_high_sales_volume` is derived from a cutoff point that determines whether a book falls into high or low sales, depending on 2021~2022 dataset from `RECOMMENDATION_*.csv` dataset
# MAGIC       - 65th percentile of `QUANTITY` distribution (aggregated view by book) serves as the cut off point
# MAGIC         - `1` denotes as high sales
# MAGIC         - `0` denotes as low sales

# COMMAND ----------

import os
import pandas as pd

# COMMAND ----------

# MAGIC %md # Load the json files that stores ISBN and google book api fetched result:
# MAGIC   - book `description` is stored there

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm_20230421_2")
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_4.json").createOrReplaceTempView("isbn_google_reomm_20230421_4")

# COMMAND ----------

# MAGIC %md # load `RECOMMENDATION_*` csv file for label variable `is_high_sales_volume`

# COMMAND ----------

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

# df_items = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_*.csv"), header=True)
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 

# df_items.createOrReplaceTempView("df_tems") # spark read
# df_reomm.createOrReplaceTempView("df_reomm") # spark read

# df_items_p = df_items.toPandas() # padnas 
df_reomm_p = df_reomm.toPandas() # padnas 


# COMMAND ----------

# MAGIC %md # a function that do the dataframe pre-processing on `RECOMMENDATION_*` csv file
# MAGIC   - drop mal-formed row
# MAGIC   - convert needed columns into an appropriate data type:
# MAGIC     - `PRICE`, `QUANTITY` and `AMOUNT`
# MAGIC   - add columns `year` , `month` and `day` from `TRANDATE`

# COMMAND ----------

def clean_recomm_df(df: pd.DataFrame) -> pd.DataFrame:

    # df_reomm_p_2 = df_reomm_p[df_reomm_p["HASHED_INVOICE_ID"].apply(lambda s: s.startswith("0x"))]
    # df_reomm_p_2 = df_reomm_p_2[~df_reomm_p_2['QUANTITY'].isnull()]
    # df_reomm_p_2 = df_reomm_p_2.drop("ISBN13", axis=1)

    # df_reomm_p_2["PRICE"] = df_reomm_p_2["PRICE"].astype(float)
    # df_reomm_p_2["QUANTITY"] = df_reomm_p_2["QUANTITY"].astype(int)
    # df_reomm_p_2["AMOUNT"] = df_reomm_p_2["AMOUNT"].astype(float)

    # df_reomm_p_2["year"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[0])
    # df_reomm_p_2["month"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[1])
    # df_reomm_p_2["day"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[2])

    ########################################################################################################################################################
    #   ref: https://adb-5911062106551859.19.azuredatabricks.net/?o=5911062106551859#notebook/3108408038812593/command/751034215087416                     #
    ########################################################################################################################################################

    df_2 = df[df["HASHED_INVOICE_ID"].apply(lambda s: s.startswith("0x"))]
    df_2 = df_2[~df_2['QUANTITY'].isnull()]
    #df_2 = df_2.drop("ISBN13", axis=1)

    df_2['ISBN13'] = df_2['ISBN13'].apply(lambda s:s.rstrip())

    df_2["PRICE"] = df_2["PRICE"].astype(float)
    df_2["QUANTITY"] = df_2["QUANTITY"].astype(int)
    df_2["AMOUNT"] = df_2["AMOUNT"].astype(float)

    df_2["year"] = df_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[0])
    df_2["month"] = df_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[1])
    df_2["day"] = df_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[2])

    return df_2

# COMMAND ----------

# MAGIC %md # further data pre-processing on `RECOMMENDATION_*` csv file and create another dataframe that stores the aggregated view by product
# MAGIC   - drop rows whose title is `Group Cash Coupon - $100`
# MAGIC   - group by the continuous variable throughout the time line by ISBN13

# COMMAND ----------

df_reomm_p_cleaned = clean_recomm_df(df = df_reomm_p)
df_reomm_p_cleaned_wo_coupon = df_reomm_p_cleaned[df_reomm_p_cleaned['TITLE'] != "Group Cash Coupon - $100"]  # exclude this item
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon.groupby("ISBN13").sum().reset_index()
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon_stat_by_isbn.sort_values(by='QUANTITY', ascending=False)

# COMMAND ----------

# MAGIC %md # obtain the 65th data point from `QUANTITY` distribution

# COMMAND ----------

q_65 = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].quantile(0.65)
# q_65

# COMMAND ----------

# MAGIC %md # prepare the dataset that has column `ISBN` and `description`

# COMMAND ----------

isbn_desc_pair = spark.sql("""
    with exploded_2 as (
    select 
        explode(items)
    from isbn_google_reomm_20230421_2
    ),
    volumeinfo_2 AS (
    select 
    col.volumeInfo.*
    FROM
        exploded_2
    ),
    exploded_4 as (
    select 
        explode(items)
    from isbn_google_reomm_20230421_4
    ),
    volumeinfo_4 AS (
    select 
    col.volumeInfo.*
    FROM
        exploded_2
    ),
    unioned AS (
    select * from volumeinfo_2
    union all 
    select * from volumeinfo_4
    ),
    isbn_image_link_pair AS (
    SELECT
        *,
        -- distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
        imageLinks.thumbnail as thumbnail
    FROM
        unioned
    )
    select 
        distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
        description
    from 
        isbn_image_link_pair
    where 
    thumbnail is not null and description is not null 
""").toPandas()

# COMMAND ----------

# MAGIC %md # check the dataframe

# COMMAND ----------

display(isbn_desc_pair)

# COMMAND ----------

isbn_desc_pair

# COMMAND ----------

# MAGIC %md # prepare the dataset that joins 1 dataframe that has `ISBN` and `description` and 1 dataframe that has aggregated `PRICE`, `QUANTITY` and `AMOUNT`

# COMMAND ----------

df_text_high_sales_label = pd.merge(isbn_desc_pair, df_reomm_p_cleaned_wo_coupon_stat_by_isbn, left_on='isbn', right_on='ISBN13', how='inner')

# COMMAND ----------

df_text_high_sales_label

# COMMAND ----------

# MAGIC %md # create binary variable `is_high_sales_volume` based on 65th percentile on `QUANTITY`

# COMMAND ----------

df_text_high_sales_label['is_high_sales_volume'] = df_text_high_sales_label['QUANTITY'].apply(lambda x: 1 if x >= q_65 else 0)

# COMMAND ----------

df_text_high_sales_label

# COMMAND ----------

df_text_high_sales_label = df_text_high_sales_label[['description', 'is_high_sales_volume']]

# COMMAND ----------

# MAGIC %md #  dataset for text model training is ready 

# COMMAND ----------

df_text_high_sales_label

# COMMAND ----------

# MAGIC %md # Text model training 
# MAGIC   - input tokenized `description` 
# MAGIC   - label or predicted variable `is_high_sales_volume`

# COMMAND ----------

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# COMMAND ----------

df_text = df_text_high_sales_label

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 1000
max_len = 150

tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(df_text['description'].values)
X = tokenizer.texts_to_sequences(df_text['description'].values)
X = pad_sequences(X, maxlen=max_len)


# COMMAND ----------

# MAGIC %md # dataset of `29094` rows with `1` and `0` label as `is_high_sales_volume` is feed to the model with 20% testing set allocated

# COMMAND ----------

y = df['is_high_sales_volume'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

model = Sequential()
model.add(Dense(64, input_dim=max_len, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


# COMMAND ----------

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# COMMAND ----------

# MAGIC %md # the ANN binary classifier on `description` got loss: 0.6244 and val_loss: 0.6339 
# MAGIC   - overfitting seems not existed

# COMMAND ----------

score = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# COMMAND ----------

# Plot accuracy over epochs
import matplotlib.pyplot as plt


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# COMMAND ----------

# Plot loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# COMMAND ----------

from sklearn.metrics import accuracy_score

# Make predictions on test data
y_pred = model.predict(X_test)

# Convert probabilities to predicted classes
y_pred_classes = np.round(y_pred)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)

print(accuracy)