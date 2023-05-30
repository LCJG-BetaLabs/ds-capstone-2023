# Databricks notebook source
import os
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
import seaborn as sns
from shutil import copyfile

# import os
# import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
# import seaborn as sns
from shutil import copyfile

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 21, 9


# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm_20230421_2")
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_4.json").createOrReplaceTempView("isbn_google_reomm_20230421_4")

# COMMAND ----------

! ls !ls /dbfs/

# COMMAND ----------

!ls /dbfs/team_j_downloaded_images/raw_20230423 | wc -l

# COMMAND ----------

# MAGIC %sql
# MAGIC with exploded_2 as (
# MAGIC   select 
# MAGIC       explode(items)
# MAGIC   from isbn_google_reomm_20230421_2
# MAGIC ),
# MAGIC volumeinfo_2 AS (
# MAGIC   select 
# MAGIC   col.volumeInfo.*
# MAGIC   FROM
# MAGIC       exploded_2
# MAGIC ),
# MAGIC exploded_4 as (
# MAGIC   select 
# MAGIC       explode(items)
# MAGIC   from isbn_google_reomm_20230421_4
# MAGIC ),
# MAGIC volumeinfo_4 AS (
# MAGIC   select 
# MAGIC   col.volumeInfo.*
# MAGIC   FROM
# MAGIC       exploded_2
# MAGIC ),
# MAGIC unioned AS (
# MAGIC   select * from volumeinfo_2
# MAGIC   union all 
# MAGIC   select * from volumeinfo_4
# MAGIC ),
# MAGIC isbn_image_link_pair AS (
# MAGIC   SELECT
# MAGIC     distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
# MAGIC     imageLinks.thumbnail as thumbnail
# MAGIC   FROM
# MAGIC     unioned
# MAGIC )
# MAGIC select 
# MAGIC   count(1)
# MAGIC from 
# MAGIC   isbn_image_link_pair
# MAGIC where 
# MAGIC   thumbnail is not null 

# COMMAND ----------

isbn_thumbnail_pairs = spark.sql("""
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
        distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
        imageLinks.thumbnail as thumbnail
    FROM
        unioned
    )
    select 
        *
    from 
    isbn_image_link_pair
    where 
    thumbnail is not null 
""").rdd.map(lambda x:{x['isbn']:x['thumbnail']}).collect()


# COMMAND ----------

isbn_thumbnail_pairs

# COMMAND ----------

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

df_items = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_*.csv"), header=True)
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 

# df_items.createOrReplaceTempView("df_items") # spark read
# df_reomm.createOrReplaceTempView("df_reomm") # spark read

df_items_p = df_items.toPandas() # padnas 
df_reomm_p = df_reomm.toPandas() # padnas 


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



# COMMAND ----------

df_reomm_p_cleaned = clean_recomm_df(df = df_reomm_p)
df_reomm_p_cleaned_wo_coupon = df_reomm_p_cleaned[df_reomm_p_cleaned['TITLE'] != "Group Cash Coupon - $100"]  # exclude this item
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon.groupby("ISBN13").sum().reset_index()
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon_stat_by_isbn.sort_values(by='QUANTITY', ascending=False)

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon_stat_by_isbn[:1000]

# COMMAND ----------



# COMMAND ----------

# sns.barplot(x='QUANTITY', y='ISBN13', data=df_reomm_p_cleaned_wo_coupon_stat_by_isbn, orient='h')
# sns.despine()  
# plt.show()

# COMMAND ----------

# MAGIC %md # the 65th percentile of `QUANTITY` is 7.0 , so we pick this as threshold

# COMMAND ----------

q_65 = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].quantile(0.65)
q_65

# COMMAND ----------

# MAGIC %md label `is_high_sales_volume` is add:
# MAGIC   - book with >= 7.0 => 1
# MAGIC   - otherwise => 0 

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon_stat_by_isbn['is_high_sales_volume'] = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].apply(lambda x: 1 if x >= q_65 else 0)

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon_stat_by_isbn

# COMMAND ----------

spark.sql("""
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
        distinct replace(replace(split(infoLink, "=")[2], "isbn:", ""),"&hl","") as isbn,
        imageLinks.thumbnail as thumbnail
    FROM
        unioned
    )
    select 
        *
    from 
    isbn_image_link_pair
    where 
    thumbnail is not null 
""").toPandas()

# COMMAND ----------

fetehed_image_isbn_list = []
dir_path = "/dbfs/team_j_downloaded_images/raw_20230423"
for filename in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, filename)):
        #print(filename)
        fetehed_image_isbn_list.append(filename.split(".")[0])

# COMMAND ----------

df_fetched_image_isbn = pd.DataFrame(fetehed_image_isbn_list)
df_fetched_image_isbn.columns = ["isbn"]
df_fetched_image_isbn

# COMMAND ----------

df_image_high_sales_label = pd.merge(df_fetched_image_isbn, df_reomm_p_cleaned_wo_coupon_stat_by_isbn, left_on='isbn', right_on='ISBN13', how='inner')

# COMMAND ----------

df_image_high_sales_label

# COMMAND ----------

# MAGIC %md ##distribution of `QUANTITY` is right-skewed

# COMMAND ----------

sns.distplot(df_image_high_sales_label['QUANTITY'])

# COMMAND ----------

def create_folder_if_not_existed(path):

    # dest_dir = "/dbfs/team_j/"

    dest_dir = path


    # Check if the directory exists
    if os.path.exists(dest_dir):
        print(f'Directory {dest_dir} exists')
    else:
        # Create the directory if it doesn't exist
        os.makedirs(dest_dir)
        print(f'Directory {dest_dir} created')

# COMMAND ----------

# dest_dir = "/dbfs/team_j/"

dest_dir = "/dbfs/team_j/image_dataset/"


# Check if the directory exists
if os.path.exists(dest_dir):
    print(f'Directory {dest_dir} exists')
else:
    # Create the directory if it doesn't exist
    os.makedirs(dest_dir)
    print(f'Directory {dest_dir} created')

# COMMAND ----------

df_image_high_sales_label_to_write = df_image_high_sales_label[["isbn", "is_high_sales_volume"]]

df_image_high_sales_label_to_write['isbn'] = df_image_high_sales_label_to_write['isbn'].apply(lambda x: x + ".jpg")
df_image_high_sales_label_to_write['isbn'] = df_image_high_sales_label_to_write['isbn'].astype(str)
df_image_high_sales_label_to_write.to_csv('/dbfs/team_j/labels2.csv', index=False, header=True, )

# COMMAND ----------

# 

# COMMAND ----------

df_image_high_sales_label[["isbn", "is_high_sales_volume"]]

# COMMAND ----------

!ls /dbfs/team_j

# COMMAND ----------

# !cat /dbfs/team_j/labels.csv

# COMMAND ----------

# MAGIC %md ## freq of label 0 and 1 ==> not balanced

# COMMAND ----------

df_image_high_sales_label[["isbn", "is_high_sales_volume"]]['is_high_sales_volume'].value_counts()

# COMMAND ----------

!cp /dbfs/team_j_downloaded_images/raw_20230423/*.jpg /dbfs/team_j/image_dataset/

# COMMAND ----------

! ls /dbfs/team_j_downloaded_images/image_Class

# COMMAND ----------

!ls /dbfs/team_j/image_dataset/ | wc -l

# COMMAND ----------

len(df_image_high_sales_label)

# COMMAND ----------

!ls /dbfs/team_j/image_dataset/

# COMMAND ----------

!ls /dbfs/team_j/

# COMMAND ----------

!ls /dbfs/team_j/image_dataset

# COMMAND ----------

# create_folder_if_not_existed("/dbfs/team_j/train")
# create_folder_if_not_existed("/dbfs/team_j/valid")

# create_folder_if_not_existed("/dbfs/team_j/train/high_sales")
# create_folder_if_not_existed("/dbfs/team_j/train/low_sales")


# COMMAND ----------

train_df['isbn'][0]

# COMMAND ----------

# MAGIC %md # model training 

# COMMAND ----------

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


# Set up the paths and constants
IMAGE_DIR = '/dbfs/team_j/image_dataset/'
CSV_FILE = '/dbfs/team_j/labels2.csv'
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# NUM_CLASSES = 3
# EPOCHS = 10

# for faster learning
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
NUM_CLASSES = 1
# EPOCHS = 5
EPOCHS = 1


# Load the CSV file into a pandas DataFrame
df = pd.read_csv(CSV_FILE)


df['isbn'] = df['isbn'].astype(str)
df['is_high_sales_volume'] = df['is_high_sales_volume'].astype(str)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Set up the data generators for the training and testing sets
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False)




# COMMAND ----------

# # Load MobileNetV2 model with pre-trained weights
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))

# # Add custom output layers for binary classification
# model = Sequential([
#     base_model,
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(NUM_CLASSES, activation='sigmoid')
# ])

# # Compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Set up ModelCheckpoint callback
# checkpoint_path = './model_checkpoint_roger_20230426'
# checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=False)

# # Train model
# history = model.fit(train_generator,
#                     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#                     epochs=EPOCHS,
#                     validation_data=test_generator,
#                     validation_steps=test_generator.samples // BATCH_SIZE,
#                     callbacks=[checkpoint])

# COMMAND ----------

# MAGIC %md ## base line model 

# COMMAND ----------

# Load MobileNetV2 model with pre-trained weights
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))

# Add custom output layers for binary classification
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # Changed from Dense(NUM_CLASSES, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up ModelCheckpoint callback
checkpoint_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=False)

# Train model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=test_generator,
                    validation_steps=test_generator.samples // BATCH_SIZE,
                    callbacks=[checkpoint])

# COMMAND ----------

# Plot training performance
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# COMMAND ----------

# # Get the training and validation loss values from the history object
# training_loss = history.history['loss']
# validation_loss = history.history['val_loss']

# # Plot the loss as a function of epoch
# plt.plot(training_loss, label='Training Loss')
# plt.plot(validation_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# COMMAND ----------

# MAGIC %md ## model with below setting:
# MAGIC - To balance the imbalanced classes in our binary classification problem, we try to use techniques such as oversampling, undersampling, or a combination of both. Here's how to oversample the minority class using the ImageDataGenerator

# COMMAND ----------

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Count the number of samples in each class
counter = Counter(train_df['is_high_sales_volume'])

# Compute the class weights
total_samples = sum(counter.values())
# class_weights = {cls: total_samples / count for cls, count in counter.items()}
class_weights = {0: 1.5091710338256312, 1: 2.9639766081871346}   #<<<< fix 

# Oversample the minority class
oversampler = RandomOverSampler(sampling_strategy='minority')
x_train_resampled, y_train_resampled = oversampler.fit_resample(train_df[['isbn']], train_df['is_high_sales_volume'])

# Create a new DataFrame with the resampled data
# train_df_resampled = pd.DataFrame({'isbn': x_train_resampled.flatten(), 'is_high_sales_volume': y_train_resampled})
train_df_resampled = pd.DataFrame({'isbn': np.ravel(x_train_resampled), 'is_high_sales_volume': y_train_resampled})

# Set up the data generators for the resampled training set
train_generator_resampled = train_datagen.flow_from_dataframe(
    dataframe=train_df_resampled,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True)

# Set up ModelCheckpoint callback
checkpoint_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426_2_with_oversample'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=False)


# Train model on resampled data
history = model.fit(train_generator_resampled,
                    steps_per_epoch=train_generator_resampled.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=test_generator,
                    validation_steps=test_generator.samples // BATCH_SIZE,
                    callbacks=[checkpoint],
                    class_weight=class_weights)

# COMMAND ----------

# MAGIC %md ## vgg16 model

# COMMAND ----------

from tensorflow.keras.applications import VGG16

# for faster learning
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
NUM_CLASSES = 1
# EPOCHS = 5
EPOCHS = 1


# Load the CSV file into a pandas DataFrame
df = pd.read_csv(CSV_FILE)


df['isbn'] = df['isbn'].astype(str)
df['is_high_sales_volume'] = df['is_high_sales_volume'].astype(str)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Set up the data generators for the training and testing sets
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)



# Count the number of samples in each class
counter = Counter(train_df['is_high_sales_volume'])

# Compute the class weights
total_samples = sum(counter.values())
# class_weights = {cls: total_samples / count for cls, count in counter.items()}
class_weights = {0: 1.5091710338256312, 1: 2.9639766081871346}   #<<<< fix 

# Oversample the minority class
oversampler = RandomOverSampler(sampling_strategy='minority')
x_train_resampled, y_train_resampled = oversampler.fit_resample(train_df[['isbn']], train_df['is_high_sales_volume'])

# Create a new DataFrame with the resampled data
# train_df_resampled = pd.DataFrame({'isbn': x_train_resampled.flatten(), 'is_high_sales_volume': y_train_resampled})
train_df_resampled = pd.DataFrame({'isbn': np.ravel(x_train_resampled), 'is_high_sales_volume': y_train_resampled})

# Set up the data generators for the resampled training set
train_generator_resampled = train_datagen.flow_from_dataframe(
    dataframe=train_df_resampled,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False)


# Load MobileNetV2 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))

# Add custom output layers for binary classification
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # Changed from Dense(NUM_CLASSES, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up ModelCheckpoint callback
checkpoint_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426_2_with_oversample_vgg16'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=False)

# Train model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=test_generator,
                    validation_steps=test_generator.samples // BATCH_SIZE,
                    callbacks=[checkpoint])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # model evaluation - confusion matrix (model_checkpoint_roger_binary_class_20230426_2_with_oversample_vgg16)

# COMMAND ----------

import numpy as np
from sklearn.metrics import confusion_matrix

# Generate predictions for the test data
y_pred = model.predict(test_generator)

# Convert the predictions to binary values
y_pred_binary = np.round(y_pred)

# Get the true labels
y_true = test_generator.classes

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred_binary)

# Print the confusion matrix
print(cm)

# COMMAND ----------

# Define the class labels
class_names = ['Class 0', 'Class 1']

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

# Set the axis labels and title
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Show the plot
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md # model training ; balanced images for both class , smallet data set

# COMMAND ----------

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


# Set up the paths and constants
IMAGE_DIR = '/dbfs/team_j/image_dataset/'
CSV_FILE = '/dbfs/team_j/labels2.csv'
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# NUM_CLASSES = 3
# EPOCHS = 10

# for faster learning
IMG_SIZE = (128, 128)
BATCH_SIZE = 30
NUM_CLASSES = 1
# EPOCHS = 5
EPOCHS = 5


# Load the CSV file into a pandas DataFrame
df = pd.read_csv(CSV_FILE)

# Filter the DataFrame to only include 3000 samples for each class
class_0_samples = df[df['is_high_sales_volume'] == 0].sample(n=3000, random_state=42)
class_1_samples = df[df['is_high_sales_volume'] == 1].sample(n=3000, random_state=42)
df = pd.concat([class_0_samples, class_1_samples])

df['isbn'] = df['isbn'].astype(str)
df['is_high_sales_volume'] = df['is_high_sales_volume'].astype(str)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Set up the data generators for the training and testing sets
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False)



# COMMAND ----------

from tensorflow.keras.applications import VGG16


# Load VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))

# Add custom output layers for binary classification
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Changed from Dense(NUM_CLASSES, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up ModelCheckpoint callback
checkpoint_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230426_6000images_vgg16'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=False)

# Train model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=test_generator,
                    validation_steps=test_generator.samples // BATCH_SIZE,
                    callbacks=[checkpoint])


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md # model training ; balanced images for both class , 7000 images for both classes data set , larger epoch , ResNet152V2

# COMMAND ----------

# for faster learning
IMG_SIZE = (128, 128)
BATCH_SIZE = 30
NUM_CLASSES = 1
# EPOCHS = 5
# EPOCHS = 6
EPOCHS = 5


# Load the CSV file into a pandas DataFrame
df = pd.read_csv(CSV_FILE)

# Filter the DataFrame to only include 3000 samples for each class
class_0_samples = df[df['is_high_sales_volume'] == 0].sample(n=7000, random_state=42)
class_1_samples = df[df['is_high_sales_volume'] == 1].sample(n=7000, random_state=42)
df = pd.concat([class_0_samples, class_1_samples])

df['isbn'] = df['isbn'].astype(str)
df['is_high_sales_volume'] = df['is_high_sales_volume'].astype(str)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Set up the data generators for the training and testing sets
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=IMAGE_DIR,
    x_col='isbn',
    y_col='is_high_sales_volume',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False)


# COMMAND ----------

from tensorflow.keras.applications import VGG16, ResNet152V2
from tensorflow.keras.layers import Dense, Flatten, Dropout


# COMMAND ----------

# Load VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))

# Add custom output layers for binary classification
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Changed from Dense(NUM_CLASSES, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up ModelCheckpoint callback
checkpoint_path = '/dbfs/team_j/model_checkpoint_roger_binary_class_20230427_2_14000images_VGG16_more_layers'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=False)

# Train model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=test_generator,
                    validation_steps=test_generator.samples // BATCH_SIZE,
                    callbacks=[checkpoint])


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

