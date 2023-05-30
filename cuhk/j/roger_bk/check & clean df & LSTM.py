# Databricks notebook source
import os
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 21, 9

container = "data1"
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"


# COMMAND ----------

# MAGIC %md # Instantiate dataframe (Spark Temp View + Pandas ) 

# COMMAND ----------

df_items = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "ITEM_MATCHING_*.csv"), header=True)
df_reomm = spark.read.format("csv").option('quote', '"').option('escape', "\"").load(os.path.join(data_path, "AML-data", "Actual_Data_2021-2022", "RECOMMENDATION_*.csv"), header=True) 

df_items.createOrReplaceTempView("df_items") # spark read
df_reomm.createOrReplaceTempView("df_reomm") # spark read

df_items_p = df_items.toPandas() # padnas 
df_reomm_p = df_reomm.toPandas() # padnas 

# COMMAND ----------

# MAGIC %md # Checker & quick clean

# COMMAND ----------

# MAGIC %md ## table: ITEM_MATCHING

# COMMAND ----------

# MAGIC %md ###### Found mal-formed PRODUCT_ID in df_items

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   PRODUCT_ID, 
# MAGIC   count(1) cnt 
# MAGIC from 
# MAGIC   df_items 
# MAGIC group by 1 
# MAGIC having cnt <> 1

# COMMAND ----------

# MAGIC %md ###### show rows with malformed `PRODUCT_ID`

# COMMAND ----------

df_items_p[~df_items_p["PRODUCT_ID"].apply(lambda s: s.startswith("9"))]

# COMMAND ----------

print(f'around {len(df_items_p[~df_items_p["PRODUCT_ID"].apply(lambda s: s.startswith("9"))]) / len(df_items_p) * 100}% with malformed PRODUCT_ID')

# COMMAND ----------

# MAGIC %md ###### FIXME: blindly drop rows with malformed PRODUCT_ID

# COMMAND ----------

df_items_p_2 = df_items_p[df_items_p["PRODUCT_ID"].apply(lambda s: s.startswith("9"))]

# COMMAND ----------

# MAGIC %md ###### check missing value occurrence in df `df_items_p_2`
# MAGIC - column `TRANSLATOR`,`BOOK_ORGPR`,`BOOK_DATE`,`BOOK_PAGES`,`BOOK_COVER` got null value
# MAGIC - see whether special handling / data imputation needed

# COMMAND ----------

sns.heatmap(df_items_p_2.isnull(), cbar=False)

# COMMAND ----------

# MAGIC %md ###### confirm that column `CREATE_DATE` got only Date information , but not HH:mm:ss 

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   distinct date_format(cast(CREATE_DATE as timestamp), "HH:mm:ss") hh_mm_ss 
# MAGIC from df_items

# COMMAND ----------

# MAGIC %md ###### checker that finds malread row (Fixed)

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   *
# MAGIC from 
# MAGIC   df_items
# MAGIC where
# MAGIC   COST = ' Book 1?"'

# COMMAND ----------

# MAGIC %md ###### found that there is row with many null rows except first 3 fields , just drop it

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   *
# MAGIC from 
# MAGIC   df_items
# MAGIC where QTY_PURCHASE is null

# COMMAND ----------

df_items_p_2 = df_items_p_2[~df_items_p_2["COST"].isnull()]

# COMMAND ----------

# MAGIC %md ###### found that some rows with `null` and `NULL` values in column `BOOK_DATE`, and thus just keep it remained as string

# COMMAND ----------

# MAGIC %sql
# MAGIC select BOOK_DATE, count(1) from df_items group by 1

# COMMAND ----------

# MAGIC %md ###### drop column `ISBN13` becasue it carries the same info. as `PRODUCT_ID`

# COMMAND ----------

df_items_p_2 = df_items_p_2.drop("ISBN13", axis=1)

# COMMAND ----------

# MAGIC %md ###### convert string into numerical fields

# COMMAND ----------

df_items_p_2["CREATE_DATE"] = pd.to_datetime(df_items_p_2["CREATE_DATE"])
df_items_p_2["COST"] = df_items_p_2["COST"].astype(float)
df_items_p_2["PRICE"] = df_items_p_2["PRICE"].astype(float)
df_items_p_2["QTY_PURCHASE"] = df_items_p_2["QTY_PURCHASE"].astype(int)
df_items_p_2["QTY_SALES"] = df_items_p_2["QTY_SALES"].astype(int)
df_items_p_2["QTY_STOCK"] = df_items_p_2["QTY_STOCK"].astype(int)
# df_items_p_2["BOOK_DATE"] = pd.to_datetime(df_items_p_2["BOOK_DATE"])

# COMMAND ----------

df_items_p_2.describe(), df_items_p_2.info()

# COMMAND ----------

df_items_p_2.head(10)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## table: RECOMMENDATION

# COMMAND ----------

# MAGIC %md ###### check missing value occurrence in df `df_reomm_p`

# COMMAND ----------

sns.heatmap(df_reomm_p.isnull(), cbar=False)

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   df_reomm
# MAGIC where 
# MAGIC  HASHED_INVOICE_ID not like '%0x%'

# COMMAND ----------

print(f'around {len(df_reomm_p[~df_reomm_p["HASHED_INVOICE_ID"].apply(lambda s: s.startswith("0x"))]) / len(df_reomm_p["HASHED_INVOICE_ID"]) * 100}% with malformed HASHED_INVOICE_ID')

# COMMAND ----------

# MAGIC %md ###### FIXME: just rougly drop rows with mal-formed `HASHED_INVOICE_ID` (please handle later)

# COMMAND ----------

df_reomm_p_2 = df_reomm_p[df_reomm_p["HASHED_INVOICE_ID"].apply(lambda s: s.startswith("0x"))]

# COMMAND ----------

df_reomm_p_2

# COMMAND ----------

# MAGIC %md ###### found that df `df_reomm_p_2` got 567 rows with null value in column `TITLE` 	`PRICE`,`QUANTITY`,`AMOUNT`,`SHOP_NO`

# COMMAND ----------

df_reomm_p_2[df_reomm_p_2['QUANTITY'].isnull()]

# COMMAND ----------

# MAGIC %md ###### FIXME: simply drop that 567 rows

# COMMAND ----------

df_reomm_p_2 = df_reomm_p_2[~df_reomm_p_2['QUANTITY'].isnull()]

# COMMAND ----------

# MAGIC %md ###### drop column `ISBN13` becasue it carries the same info. as `PRODUCT`

# COMMAND ----------

df_reomm_p_2 = df_reomm_p_2.drop("ISBN13", axis=1)

# COMMAND ----------

# MAGIC %md ###### convert string into numerical fields

# COMMAND ----------

df_reomm_p_2["PRICE"] = df_reomm_p_2["PRICE"].astype(float)
df_reomm_p_2["QUANTITY"] = df_reomm_p_2["QUANTITY"].astype(int)
df_reomm_p_2["AMOUNT"] = df_reomm_p_2["AMOUNT"].astype(float)

# COMMAND ----------

df_items_p_2

# COMMAND ----------

df_reomm_p_2

# COMMAND ----------

# MAGIC %md # Inner join two df

# COMMAND ----------

merged_df = pd.merge(df_reomm_p_2, df_items_p_2, left_on='PRODUCT', right_on='PRODUCT_ID', how='inner')

# COMMAND ----------

# MAGIC %md ###### FIXME: have to see whether there is another approach that could restore the majority of dropped rows after inner join 
# MAGIC   - shape: 54899 rows × 28 columns right now

# COMMAND ----------

# MAGIC %md ###### drop duplicate columns and rename

# COMMAND ----------

merged_df = merged_df.drop([
    "TITLE_y",
    "PRICE_y"
], axis=1)

merged_df = merged_df.rename(columns={'TITLE_x': 'TITLE', 'PRICE_x': 'PRICE'})

# COMMAND ----------

merged_df.head()

# COMMAND ----------

# MAGIC %md # visualize (brainstorm)

# COMMAND ----------

sns.pairplot(merged_df)

# COMMAND ----------

sns.countplot(x='TITLE',data=merged_df) # just for checking

# COMMAND ----------

# MAGIC %md # check matched row count between two provided tables
# MAGIC
# MAGIC - low matched rate on ISBN code 
# MAGIC   - so, we don't use `ITEM` table

# COMMAND ----------

# MAGIC %sql
# MAGIC with joined AS (
# MAGIC   select 
# MAGIC     *
# MAGIC   from
# MAGIC     df_reomm r
# MAGIC   left join
# MAGIC     df_items i on trim(i.ISBN13) = trim(r.ISBN13)
# MAGIC ),
# MAGIC matched_count AS (
# MAGIC   select 
# MAGIC     count(1) matched_cnt
# MAGIC   from 
# MAGIC     joined
# MAGIC   WHERE
# MAGIC     PRD_CATEGORY is not null
# MAGIC ),
# MAGIC total_count AS (
# MAGIC   select count(1) as total_cnt from df_reomm
# MAGIC )
# MAGIC select (select matched_cnt from matched_count) / (select total_cnt from total_count) * 100 as matched_rate_in_percent

# COMMAND ----------

# MAGIC %md # model preaparation 

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import datetime

# COMMAND ----------

# MAGIC %md ###### ONLY df `df_reomm_p_2` is used in this section

# COMMAND ----------

df_reomm_p_2["PRODUCT"].nunique()

# COMMAND ----------

# MAGIC %md ###### obtain year, month and day column

# COMMAND ----------

df_reomm_p_2["year"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[0])
df_reomm_p_2["month"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[1])
df_reomm_p_2["day"] = df_reomm_p_2['TRANDATE'].apply(lambda x: x.split(" ")[0].split("-")[2])

# COMMAND ----------

df_reomm_p_2.head()

# COMMAND ----------

amount_over_time = df_reomm_p_2[[
    "year",
    "month",
    "day",
    "AMOUNT"
]].groupby([
    "year",
    "month",
    "day",
]).sum().reset_index()

# COMMAND ----------

# MAGIC %md ###### plot trend

# COMMAND ----------

amount_over_time['date'] = pd.to_datetime(amount_over_time[['year', 'month', 'day']])

# set datetime column as index
amount_over_time.set_index('date', inplace=True)

# plot time series chart
amount_over_time.plot(kind='line', y='AMOUNT')

# set plot title and axis labels
plt.title('Amount over Time')
plt.xlabel('Date')
plt.ylabel('Amount')

# display plot
plt.show()


# COMMAND ----------

# wider plot width 
plt.figure(figsize=(24, 6))

# COMMAND ----------

# calculate moving averages and add to DataFrame
amount_over_time['MA_10'] = amount_over_time['AMOUNT'].rolling(window=10).mean()
amount_over_time['MA_20'] = amount_over_time['AMOUNT'].rolling(window=20).mean()

# calculate exponential moving average and add to DataFrame
amount_over_time['EMA'] = amount_over_time['AMOUNT'].ewm(span=10, adjust=False).mean()

# calculate Bollinger bands and add to DataFrame
amount_over_time['20_day_std'] = amount_over_time['AMOUNT'].rolling(window=20).std()
amount_over_time['upper_band'] = amount_over_time['MA_20'] + (2 * amount_over_time['20_day_std'])
amount_over_time['lower_band'] = amount_over_time['MA_20'] - (2 * amount_over_time['20_day_std'])

# plot time series chart with moving averages and Bollinger bands
plt.plot(amount_over_time.index, amount_over_time['AMOUNT'], label='Amount')
plt.plot(amount_over_time.index, amount_over_time['MA_10'], label='10-day MA')
plt.plot(amount_over_time.index, amount_over_time['MA_20'], label='20-day MA')
plt.plot(amount_over_time.index, amount_over_time['EMA'], label='EMA')
plt.plot(amount_over_time.index, amount_over_time['upper_band'], 'r--', label='Upper Bollinger Band')
plt.plot(amount_over_time.index, amount_over_time['lower_band'], 'r--', label='Lower Bollinger Band')

# set plot title and axis labels
plt.title('Amount over Time')
plt.xlabel('Date')
plt.ylabel('Amount')

# add legend
plt.legend()

# display plot
plt.show()


# COMMAND ----------



# COMMAND ----------

amount_over_time["year"] = amount_over_time["year"].astype(int)
amount_over_time["month"] = amount_over_time["month"].astype(int)
amount_over_time["day"] = amount_over_time["day"].astype(int)


# COMMAND ----------

# MAGIC %md fit model 

# COMMAND ----------

amount_over_time_1 = amount_over_time[[
    "year",
    "month",
    "day",
    "AMOUNT" 	
]]
    
    
# Scale all columns in the dataframe to the range of 0 to 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(amount_over_time_1.values)

# Create a new dataframe with scaled values
scaled_df = pd.DataFrame(scaled_data, columns=amount_over_time_1.columns)

# Define the number of time steps and features
n_steps = 3  # Use the "year", "month", and "day" columns as input features
n_features = 1  # Use the "AMOUNT" column as output feature

# Perform train-test split with 0.8
train_size = int(len(scaled_df) * 0.8)
train_df = scaled_df[:train_size]
test_df = scaled_df[train_size:]

# Prepare the training data
X_train, y_train = [], []
for i in range(n_steps, len(train_df)):
    X_train.append(train_df.iloc[i-n_steps:i][['AMOUNT']].values)
    y_train.append(train_df.iloc[i]['AMOUNT'])
X_train, y_train = np.array(X_train), np.array(y_train)

# Prepare the testing data
X_test, y_test = [], []
for i in range(n_steps, len(test_df)):
    X_test.append(test_df.iloc[i-n_steps:i][['AMOUNT']].values)
    y_test.append(test_df.iloc[i]['AMOUNT'])
X_test, y_test = np.array(X_test), np.array(y_test)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


# COMMAND ----------

train_df

# COMMAND ----------

test_df

# COMMAND ----------

train_df.iloc[i-n_steps:i][['AMOUNT']].values

# COMMAND ----------

train_df.iloc[i]['AMOUNT']

# COMMAND ----------



# COMMAND ----------

# scaler = MinMaxScaler(feature_range=(0, 1))
# amount_over_time['AMOUNT_normalized'] = scaler.fit_transform(amount_over_time[['AMOUNT']])

# # Split the data into training and testing sets
# train_size = int(len(amount_over_time) * 0.8)
# train_df = amount_over_time[:train_size]
# test_df = amount_over_time[train_size:]

# # Define the number of time steps and features
# n_steps = 3  # Use the "year", "month", and "day" columns as input features
# n_features = 1  # Use the "AMOUNT_normalized" column as output feature

# # Prepare the training data
# X_train, y_train = [], []
# for i in range(n_steps, len(train_df)):
#     X_train.append(train_df.iloc[i-n_steps:i][['AMOUNT_normalized']].values)
#     y_train.append(train_df.iloc[i]['AMOUNT_normalized'])
# X_train, y_train = np.array(X_train), np.array(y_train)

# # Prepare the testing data
# X_test, y_test = [], []
# for i in range(n_steps, len(test_df)):
#     X_test.append(test_df.iloc[i-n_steps:i][['AMOUNT_normalized']].values)
#     y_test.append(test_df.iloc[i]['AMOUNT_normalized'])
# X_test, y_test = np.array(X_test), np.array(y_test)

# # Define the LSTM model
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# # Train the model
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


# COMMAND ----------

# list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# COMMAND ----------

# MAGIC %md ###### hyperparameter tuning with grid search

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# COMMAND ----------

# Define a function to create the model for GridSearchCV
def create_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Create a KerasRegressor wrapper for use with GridSearchCV
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the hyperparameters to tune
param_grid = {
    'batch_size': [32, 64],
    'epochs': [50, 100],
#     'optimizer': ['adam']
}

# Create a GridSearchCV object
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

# Fit the GridSearchCV object to the training data
grid_result = grid.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_result.best_params_)
print("Best score:", grid_result.best_score_)


# COMMAND ----------

# y_train

# COMMAND ----------

# Make predictions on training and testing data
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate training and testing set loss
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

# Calculate training and testing set accuracy
train_acc = 1 - train_loss
test_acc = 1 - test_loss

# Plot performance charts
plt.plot(y_train, label='Actual (Training)')
plt.plot(train_preds, label='Predicted (Training)')
plt.title('Training Set Performance')
plt.xlabel('Time')
plt.ylabel('Normalized Amount')
plt.legend()
plt.show()

plt.plot(y_test, label='Actual (Testing)')
plt.plot(test_preds, label='Predicted (Testing)')
plt.title('Testing Set Performance')
plt.xlabel('Time')
plt.ylabel('Normalized Amount')
plt.legend()
plt.show()

print('Training Loss:', train_loss)
print('Training Accuracy:', train_acc)
print('Testing Loss:', test_loss)
print('Testing Accuracy:', test_acc)

# COMMAND ----------

# MAGIC %md ###### Use the model to predict the AMOUNT for 2023:

# COMMAND ----------

X_test

# COMMAND ----------

start_of_year = datetime.datetime(datetime.datetime.now().year, 1, 1).strftime("%Y-%m-%d")
today = datetime.datetime.now().strftime("%Y-%m-%d")

date_range = pd.date_range(start=start_of_year, end=today)

df_2023 = pd.DataFrame(date_range)

df_2023["year"] = df_2023[0].apply(lambda x: str(x).split(" ")[0].split("-")[0])
df_2023["month"] = df_2023[0].apply(lambda x: str(x).split(" ")[0].split("-")[1])
df_2023["day"] = df_2023[0].apply(lambda x: str(x).split(" ")[0].split("-")[2])

df_2023 = df_2023[[
    "year",
    "month",
    "day"
]]

# COMMAND ----------

df_2023.head(5)

# COMMAND ----------

input_data = df_2023.values.reshape(-1, 1)
# Scale the input data using the same scaler as before
input_data = scaler.transform(input_data)
# Reshape the input data to be 3D for LSTM input
input_data = input_data.reshape(1, len(df_2023), 1)
# Predict the `AMOUNT` for 2023
# predicted_amount = scaler.inverse_transform(model.predict(input_data))[0][0]


# COMMAND ----------

df_2023[0]

# COMMAND ----------

