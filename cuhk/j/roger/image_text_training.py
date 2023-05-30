# Databricks notebook source
# MAGIC %md # this model only takes into account the `description` of a book as the model input and the label `is_high_sales_volume` as 

# COMMAND ----------

import os
import pandas as pd

# COMMAND ----------

spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_2.json").createOrReplaceTempView("isbn_google_reomm_20230421_2")
spark.read.json("dbfs:/dbfs/isbn_google_reomm_20230421_4.json").createOrReplaceTempView("isbn_google_reomm_20230421_4")

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
df_reomm_p_cleaned_wo_coupon = df_reomm_p_cleaned[df_reomm_p_cleaned['TITLE'] != "Group Cash Coupon - $100"]  # exclude this item
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon.groupby("ISBN13").sum().reset_index()
df_reomm_p_cleaned_wo_coupon_stat_by_isbn = df_reomm_p_cleaned_wo_coupon_stat_by_isbn.sort_values(by='QUANTITY', ascending=False)

# COMMAND ----------

q_65 = df_reomm_p_cleaned_wo_coupon_stat_by_isbn['QUANTITY'].quantile(0.65)
# q_65

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

display(isbn_desc_pair)

# COMMAND ----------

isbn_desc_pair

# COMMAND ----------

df_text_high_sales_label = pd.merge(isbn_desc_pair, df_reomm_p_cleaned_wo_coupon_stat_by_isbn, left_on='isbn', right_on='ISBN13', how='inner')

# COMMAND ----------

df_text_high_sales_label

# COMMAND ----------

df_text_high_sales_label['is_high_sales_volume'] = df_text_high_sales_label['QUANTITY'].apply(lambda x: 1 if x >= q_65 else 0)

# COMMAND ----------

df_text_high_sales_label

# COMMAND ----------

df_text_high_sales_label = df_text_high_sales_label[['description', 'is_high_sales_volume']]

# COMMAND ----------

df_text_high_sales_label

# COMMAND ----------

# MAGIC %md # model training - text only 

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

# COMMAND ----------

# MAGIC %md # text model (v2)

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataframe
df = df_text_high_sales_label

# Sample 3000 for label 1 and 3000 for label 0
label_1_df = df[df["is_high_sales_volume"] == 1].sample(n=5000, random_state=42)
label_0_df = df[df["is_high_sales_volume"] == 0].sample(n=5000, random_state=42)
df = pd.concat([label_1_df, label_0_df])

# Extract text and label
text = df["description"]
label = df["is_high_sales_volume"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.2, random_state=42)

# Tokenize and pad the text
max_words = 1000
max_len = 150
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(X_train.values)
X_train = tokenizer.texts_to_sequences(X_train.values)
X_train = pad_sequences(X_train, maxlen=max_len)

tokenizer.fit_on_texts(X_test.values)
X_test = tokenizer.texts_to_sequences(X_test.values)
X_test = pad_sequences(X_test, maxlen=max_len)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=32, input_length=max_len))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Define checkpoint path
checkpoint_path = "/dbfs/team_j/text_model_2_roger_20230427_lstm"

# Define callbacks
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint])

# COMMAND ----------

! ls "/dbfs/team_j/"

# COMMAND ----------

# MAGIC %md # text model v3

# COMMAND ----------

# !pip install -U gensim

# COMMAND ----------

# !pip install --upgrade numpy==1.20.3

# COMMAND ----------



# COMMAND ----------

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM
# from tensorflow.keras.optimizers import Adam
# # from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.callbacks import ModelCheckpoint

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import hstack


# COMMAND ----------


# # Load the dataframe
# df = df_text_high_sales_label

# # Sample 5000 for label 1 and 5000 for label 0
# label_1_df = df[df["is_high_sales_volume"] == 1].sample(n=5000, random_state=42)
# label_0_df = df[df["is_high_sales_volume"] == 0].sample(n=5000, random_state=41)
# df = pd.concat([label_1_df, label_0_df])

# # Extract text and label
# text = df["description"]
# label = df["is_high_sales_volume"]

# # Train test split
# X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.2, random_state=42)

# # Vectorize the text with tfidf
# max_words = 1000
# max_len = 150
# vectorizer = TfidfVectorizer(max_features=max_words, stop_words='english')
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)


# # Combine the tfidf vectors with the sequence of word embeddings
# X_train_combined = hstack([X_train_tfidf, y_train])
# X_test_combined = hstack([X_test_tfidf, y_test])


# # Build the model
# model = Sequential()
# model.add(LSTM(64, input_shape=(max_len, 100)))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# # Define checkpoint path
# checkpoint_path = "/dbfs/team_j/text_model"

# # Define callbacks
# checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# # Train the model
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint])


# COMMAND ----------



# COMMAND ----------

# MAGIC %md # model training (text + image)

# COMMAND ----------

from tensorflow.keras.applications import VGG16

# COMMAND ----------

# import pandas as pd
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# # from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.callbacks import ModelCheckpoint
# import matplotlib.pyplot as plt


# # Set up the paths and constants
# IMAGE_DIR = '/dbfs/team_j/image_dataset/'
# CSV_FILE = '/dbfs/team_j/labels2.csv'
# # IMG_SIZE = (224, 224)
# # BATCH_SIZE = 32
# # NUM_CLASSES = 3
# # EPOCHS = 10

# # for faster learning
# IMG_SIZE = (128, 128)
# BATCH_SIZE = 30
# NUM_CLASSES = 1
# # EPOCHS = 5
# EPOCHS = 5


# # Load the CSV file into a pandas DataFrame
# df = pd.read_csv(CSV_FILE)

# # Filter the DataFrame to only include 3000 samples for each class
# class_0_samples = df[df['is_high_sales_volume'] == 0].sample(n=3000, random_state=42)
# class_1_samples = df[df['is_high_sales_volume'] == 1].sample(n=3000, random_state=42)
# df = pd.concat([class_0_samples, class_1_samples])

# df['isbn'] = df['isbn'].astype(str)
# df['is_high_sales_volume'] = df['is_high_sales_volume'].astype(str)

# # Split the data into training and testing sets
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# # Set up the data generators for the training and testing sets
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    rotation_range=20,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_dataframe(
#     dataframe=train_df,
#     directory=IMAGE_DIR,
#     x_col='isbn',
#     y_col='is_high_sales_volume',
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     shuffle=True)

# test_generator = test_datagen.flow_from_dataframe(
#     dataframe=test_df,
#     directory=IMAGE_DIR,
#     x_col='isbn',
#     y_col='is_high_sales_volume',
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     shuffle=False)



# COMMAND ----------

# base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))

# # Freeze the pre-trained weights
# for layer in base_model.layers:
#     layer.trainable = False

# # Define image input
# image_input = Input(shape=IMG_SIZE+(3,))

# # Pass image input through base_model
# x = base_model(image_input)

# # Flatten the output of base_model
# x = Flatten()(x)

# # Add custom dense layers for binary classification
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(32, activation='relu')(x)
# output = Dense(1, activation='sigmoid')(x) # Changed from Dense(NUM_CLASSES, activation='sigmoid')

# # Create the model
# image_model = Model(inputs=image_input, outputs=output)

# COMMAND ----------



# COMMAND ----------

# # Text model
# text_input = Input(shape=(MAX_LEN,))
# y = Dense(64, activation='relu')(text_input)
# y = Dropout(0.2)(y)
# y = Dense(32, activation='relu')(y)
# y = Dropout(0.2)(y)
# text_output = Dense(1, activation='sigmoid')(y)
# text_model = Model(inputs=text_input, outputs=text_output)

# # Concatenate image and text models
# combined = concatenate([x, text_model.output])

# # Add custom output layers for binary classification
# output = Dense(32, activation='relu')(combined)
# output = Dense(1, activation='sigmoid')(output)


# COMMAND ----------

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input

# def build_text_model(max_len=150):
#     model = Sequential()
#     model.add(Input(shape=(max_len,)))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     return model

# COMMAND ----------

text_model = build_text_model()

# COMMAND ----------


from keras.layers import concatenate

# Concatenate the outputs of the two models
merged = concatenate([image_model.output, text_model.output])

# Define final output layer
output = Dense(1, activation='sigmoid')(merged)


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

