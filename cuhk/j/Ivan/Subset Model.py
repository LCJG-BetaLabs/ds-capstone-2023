# Databricks notebook source
import matplotlib.pyplot as plotter_lib
import numpy as np
import PIL as image_lib
import tensorflow as tflow
from tensorflow.keras.layers import Flatten
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# COMMAND ----------

img_height,img_width=190,128
batch_size=32
train_ds = tflow.keras.preprocessing.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class_Subset/",
  validation_split=0.2,
  subset="training",
  seed=123,

label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class_Subset/",
  validation_split=0.2,
  subset="validation",
  seed=123,
label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

resnet_model = Sequential()

pretrained_model_for_demo= tflow.keras.applications.ResNet50(include_top=False,

                   input_shape=(190,128,3),

                   pooling='avg',classes=3,

                   weights='imagenet'
                   #weights= None
                   )

for each_layer in pretrained_model_for_demo.layers:
        each_layer.trainable=False

resnet_model.add(pretrained_model_for_demo)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
#resnet_model.add(Dense(256, activation='relu'))
#resnet_model.add(Dense(128, activation='relu'))
resnet_model.add(Dense(3, activation='softmax'))



# COMMAND ----------

resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

history = resnet_model.fit(train_ds, validation_data=validation_ds, epochs=10)



# COMMAND ----------

plotter_lib.figure(figsize=(8, 8))

epochs_range= range(10)

plotter_lib.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")

plotter_lib.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")

plotter_lib.axis(ymin=0.1,ymax=1)

plotter_lib.grid()

plotter_lib.title('Model Accuracy')

plotter_lib.ylabel('Accuracy')

plotter_lib.xlabel('Epochs')

plotter_lib.legend(['train', 'validation'])

# COMMAND ----------

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# COMMAND ----------

demo_resnet_model.save('/dbfs/team_j/model_checkpoint_ivan_three_class_Subset_Trial.h5')

# COMMAND ----------

import glob
glob.glob(f"/dbfs/team_j_downloaded_images/image_Class_Subset/*",)

# COMMAND ----------

import os

names = os.listdir('/dbfs/team_j_downloaded_images/image_Class_Subset/LowSales')

names[:5]
names_Med = os.listdir('/dbfs/team_j_downloaded_images/image_Class_Subset/MediumSales')

names[:5]
names_Top = os.listdir('/dbfs/team_j_downloaded_images/image_Class_Subset/TopSales')

names[:5]


# COMMAND ----------

import pandas as pd

names_df_LowSales = pd.DataFrame({'Name':[x.split('.')[0] for x in names], 
                         'Path':['/dbfs/team_j_downloaded_images/image_Class_Subset/LowSales/' + x for x in names],
                         'Class':'Low Sales'}
                         )


names_df_MedSales = pd.DataFrame({'Name':[x.split('.')[0] for x in names_Med], 
                         'Path':['/dbfs/team_j_downloaded_images/image_Class_Subset/MediumSales/' + x for x in names_Med],
                         'Class':'Med Sales'
                         }
                         
                         )

names_df_TopSales = pd.DataFrame({'Name':[x.split('.')[0] for x in names_Top], 
                         'Path':['/dbfs/team_j_downloaded_images/image_Class_Subset/TopSales/' + x for x in names_Top],
                         'Class':'Top Sales'
                         }
                         
                         )                       
                         

# COMMAND ----------

names_df_TopSales

# COMMAND ----------

import pandas as pd
df_con = pd.concat([names_df_LowSales, names_df_MedSales, names_df_TopSales])


# COMMAND ----------

df_con = df_con.sample(frac=1).reset_index(drop = True)

# COMMAND ----------

df_con

# COMMAND ----------

import os
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
import seaborn as sns
from shutil import copyfile
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
import re
from sklearn.utils import shuffle
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 21, 9


# COMMAND ----------

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df_items = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_*.csv"), header=True)
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 

# df_items.createOrReplaceTempView("df_items") # spark read
# df_reomm.createOrReplaceTempView("df_reomm") # spark read

df_items_p = df_items.toPandas() # padnas 
df_reomm_p = df_reomm.toPandas() # padnas 

# external dataset from googleapi books
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm")


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

df_reomm_p_cleaned = clean_recomm_df(df = df_reomm_p)

# COMMAND ----------

df_reomm_p_cleaned['Date'] = df_reomm_p_cleaned['year'] +'-'+ df_reomm_p_cleaned['month']+'-'+ df_reomm_p_cleaned['day']
df_reomm_p_cleaned = df_reomm_p_cleaned.drop_duplicates()

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded as (
# MAGIC select 
# MAGIC     explode(items)
# MAGIC from isbn_google_reomm
# MAGIC ),
# MAGIC volumeinfo AS (
# MAGIC select 
# MAGIC     col.volumeInfo.*
# MAGIC FROM
# MAGIC     exploded
# MAGIC )
# MAGIC select 
# MAGIC lower(publisher)
# MAGIC from 
# MAGIC volumeinfo
# MAGIC where imageLinks.thumbnail is not null and categories[0] is not null 

# COMMAND ----------

df_isbn_google_reomm_cleaned = spark.sql("""
    with exploded as (
    select 
        explode(items)
    from isbn_google_reomm
    ),
    volumeinfo AS (
    select 
        col.volumeInfo.*
    FROM
        exploded
    )
    select 
    lower(authors[0]) as authors,
    lower(publisher) as publisher,
    lower(categories[0]) as categories,
    lower(description) as description,
    imageLinks.thumbnail,
    lower(title) as title,
    infoLink,
    replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn
    from 
    volumeinfo
    where imageLinks.thumbnail is not null and categories[0] is not null 
""").toPandas()

df_isbn_google_reomm_cleaned = df_isbn_google_reomm_cleaned.drop_duplicates()

# COMMAND ----------

df_Grouped_recom = df_reomm_p_cleaned.groupby(
                    ['Date', 'year', 'month', 'day','PRODUCT','ISBN13','TITLE','SHOP_NO']
                    ).agg(
                        {
                            'QUANTITY':sum,    # Sum sales quantity
                            'PRICE': 'first'  # get the first price per group
                        }
                    ).reset_index(drop=False)

# COMMAND ----------

df_Grouped_recom_ISBN = df_reomm_p_cleaned.groupby(
                    ['ISBN13']
                    ).agg(
                        {
                            'QUANTITY':sum    # Sum sales quantity
                        
                        }
                    ).reset_index(drop=False)

# COMMAND ----------

x = df_Grouped_recom_ISBN.groupby(['QUANTITY']).size().reset_index(name='counts')

x['Selling_Qty'] = np.nan

for i in range (0, len(x)):
    if x['QUANTITY'].iloc[i] > 5 and x['QUANTITY'].iloc[i] < 25:
        x['Selling_Qty'].iloc[i] = '6 - 24'
    elif x['QUANTITY'].iloc[i] >=25:
        x['Selling_Qty'].iloc[i] = '25+'
    else:
        x['Selling_Qty'].iloc[i] = int(x['QUANTITY'].iloc[i])


# COMMAND ----------

y = x.groupby(
                    ['Selling_Qty']
                    ).agg(
                        {
                            'counts':sum    # Sum sales quantity
                        
                        }
                    ).reset_index(drop=False)

# COMMAND ----------

y

# COMMAND ----------

y['cumperc'] = y['counts'].cumsum()/y['counts'].sum()*100


# COMMAND ----------

y

# COMMAND ----------

custom_dict = {1: 0, 2: 1, 3: 2, 4 :3,5:4, '6 - 24':5,'25+':6}
y['rank'] = y['Selling_Qty'].map(custom_dict)

# COMMAND ----------

y.sort_values(by=['rank'], inplace = True)

# COMMAND ----------

y['cumperc'] = y['counts'].cumsum()/y['counts'].sum()*100

# COMMAND ----------

y

# COMMAND ----------

y.drop(labels=['rank'],axis=1)

# COMMAND ----------

#sort DataFrame by count descending
y = y.sort_values(by='count', ascending=False)

#add column to display cumulative percentage
df['cumperc'] = df['count'].cumsum()/df['count'].sum()*100

# COMMAND ----------

y = y.set_index('Selling_Qty')

# COMMAND ----------

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

#define aesthetics for plot
color1 = 'steelblue'
color2 = 'red'
line_size = 4

#create basic bar plot
fig, ax = plt.subplots()
ax.bar(y.index, y['counts'], color=color1)

#add cumulative percentage line to plot
ax2 = ax.twinx()
ax2.plot(y.index, y['cumperc'], color=color2, marker="D", ms=line_size)
ax2.yaxis.set_major_formatter(PercentFormatter())

#specify axis colors
ax.tick_params(axis='y', colors=color1)
ax2.tick_params(axis='y', colors=color2)

#display Pareto chart
plt.show()

# COMMAND ----------

len(df_Grouped_recom_ISBN)

# COMMAND ----------

print (df_Grouped_recom_ISBN['QUANTITY'].quantile(q=0.6))
print (df_Grouped_recom_ISBN['QUANTITY'].quantile(q=0.9))

# COMMAND ----------

print(len(df_Grouped_recom_ISBN[df_Grouped_recom_ISBN['QUANTITY'] == 1]))

# COMMAND ----------

df_merged = pd.merge(df_Grouped_recom,df_isbn_google_reomm_cleaned, left_on = 'ISBN13', right_on = 'isbn', how = 'inner')

# COMMAND ----------

len(df_merged)

# COMMAND ----------

df_merged.columns 

# COMMAND ----------

df_merged.head()

# COMMAND ----------

df_unique = df_merged[['ISBN13','authors', 'publisher', 'categories']].drop_duplicates()

# COMMAND ----------


merged_df = pd.merge(left=df_con, right=df_unique, left_on='Name', right_on = 'ISBN13', how='inner')

merged_df = merged_df.drop(['ISBN13'], axis=1)

# COMMAND ----------

merged_df.columns

# COMMAND ----------

#Drop is any value is null
merged_df_clean = merged_df.dropna()

# COMMAND ----------

one_hot_authors = pd.get_dummies(merged_df_clean['authors'])
one_hot_publisher = pd.get_dummies(merged_df_clean['publisher'])
one_hot_categories = pd.get_dummies(merged_df_clean['categories'])
merged_onehot = pd.concat([merged_df_clean,one_hot_authors,one_hot_publisher,one_hot_categories], axis=1)


# COMMAND ----------

merged_onehot = pd.concat([merged_df_clean,one_hot_authors,one_hot_publisher,one_hot_categories], axis=1)

# COMMAND ----------

merged_onehot.head()

# COMMAND ----------

merged_onehot = merged_onehot.drop(columns=['authors', 'publisher','categories','Path'])

# COMMAND ----------

merged_onehot.head()

# COMMAND ----------

#!pip install opencv-python

# COMMAND ----------

merged_onehot.head()

# COMMAND ----------

merged_onehot['Class'].unique()

# COMMAND ----------

merged_onehot['Class_R'] = np.nan

# COMMAND ----------

merged_onehot['Class'].unique()

# COMMAND ----------

Value = {'Top Sales': 1, 'Med Sales':2, 'Low Sales': 3 }

# COMMAND ----------

merged_onehot['Class_R'] = merged_onehot['Class'].map(Value)

# COMMAND ----------

merged_onehot.head()

# COMMAND ----------

merged_onehot['Class_R'].unique()

# COMMAND ----------

merged_onehot_x = merged_onehot.drop(columns=['Name', 'Class'])
merged_onehot_y = merged_onehot[['Class_R']]

# COMMAND ----------

merged_onehot_y.to_numpy()

# COMMAND ----------

from sklearn.model_selection import train_test_split
X, y = merged_onehot_x.to_numpy(),merged_onehot_y.to_numpy()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

model = Sequential()
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

# COMMAND ----------

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# COMMAND ----------

model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_test, y_test))

# COMMAND ----------

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

