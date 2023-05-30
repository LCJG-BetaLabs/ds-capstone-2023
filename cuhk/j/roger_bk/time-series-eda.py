# Databricks notebook source
# MAGIC %md # setup 

# COMMAND ----------

import os
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings
import seaborn as sns
from shutil import copyfile
import matplotlib.dates as mdates
import statsmodels.api as sm

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 21, 9

# COMMAND ----------

# set display options to show all columns and rows and the full text
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# COMMAND ----------

# MAGIC %md # dataframe preparation

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

# MAGIC %md # datafram transformation

# COMMAND ----------

# MAGIC %md #### `RECOMMENDATION` dataset

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

    df_2['TITLE'] = df_2['TITLE'].apply(lambda x:x.rstrip())

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

df_reomm_p_cleaned.head(5)

# COMMAND ----------

df_title_count = pd.DataFrame(df_reomm_p_cleaned['TITLE'].value_counts()).reset_index()
df_title_count = df_title_count.rename(columns={'index': 'title', 'TITLE':'count'})
df_title_count = df_title_count.sort_values(by="count",ascending=False)

# COMMAND ----------

# MAGIC %md # Top 50 Book title by count

# COMMAND ----------

import seaborn as sns
import pandas as pd

# only consider top N books by count
n = 50
df_title_count_top_n = df_title_count.sort_values(by='count', ascending=False).head(n)

# # sort the dataframe by count in descending order
# df_title_count = df_title_count.sort_values(by='count', ascending=False)

# create a horizontal bar chart with seaborn
sns.set_style('whitegrid')

# label the weird title `Group Cash Coupon - $100`
colors = ['coral' if ( x == 'Group Cash Coupon - $100' ) else 'skyblue' for x in df_title_count['title']]

sns.barplot(x='count', y='title', data=df_title_count_top_n, palette=colors)

# set the chart title and axis labels
plt.title(f'Top {n} Book Title by Count')
plt.xlabel('Count')
plt.ylabel('Title')

# display the chart
plt.show()


# COMMAND ----------

# MAGIC %md #### discard title  = `Group Cash Coupon - $100`

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon = df_reomm_p_cleaned[df_reomm_p_cleaned['TITLE'] != "Group Cash Coupon - $100"]
df_reomm_p_cleaned_w_coupon = df_reomm_p_cleaned[df_reomm_p_cleaned['TITLE'] == "Group Cash Coupon - $100"]

# COMMAND ----------

df_reomm_p_cleaned_wo_coupon.head()

# COMMAND ----------

amount_over_time = df_reomm_p_cleaned_wo_coupon[[
    "year",
    "month",
    "day",
    "AMOUNT"
]].groupby([
    "year",
    "month",
    "day",
]).sum().reset_index()
amount_over_time['date'] = pd.to_datetime(amount_over_time[['year', 'month', 'day']])


# COMMAND ----------

amount_over_time.head()

# COMMAND ----------

# MAGIC %md # trend

# COMMAND ----------

# MAGIC %md #### non coupon sales

# COMMAND ----------

pivot_table = amount_over_time.pivot_table(index='month', columns='year', values='AMOUNT')
pivot_table.plot(kind='line')
plt.title('MoM trend (2021 & 2022)')
plt.show()

# COMMAND ----------

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data
ax.plot(amount_over_time['date'], amount_over_time['AMOUNT'])

# Calculate the moving averages
ma_30 = amount_over_time['AMOUNT'].rolling(window=30).mean()
ema_30 = amount_over_time['AMOUNT'].ewm(span=30, adjust=False).mean()

# Calculate the bollinger bands
ma_20 = amount_over_time['AMOUNT'].rolling(window=20).mean()
std_20 = amount_over_time['AMOUNT'].rolling(window=20).std()
upper_band = ma_20 + (2 * std_20)
lower_band = ma_20 - (2 * std_20)

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data and moving averages
ax.plot(amount_over_time['date'], amount_over_time['AMOUNT'], label='Amount')
ax.plot(amount_over_time['date'], ma_30, label='MA(30)')
ax.plot(amount_over_time['date'], ema_30, label='EMA(30)')

# Plot the bollinger bands
ax.plot(amount_over_time['date'], upper_band, label='Upper Band')
ax.plot(amount_over_time['date'], lower_band, label='Lower Band')

# Add a legend
ax.legend()


# Set the title and axis labels
ax.set_title('Amount over Time (2021-2022)')
ax.set_xlabel('Date')
ax.set_ylabel('Amount')

# Show the plot
plt.show()


# COMMAND ----------

# MAGIC %md ####
# MAGIC - look stationary and no trend over the months

# COMMAND ----------

# MAGIC %md #### usage on `Group Cash Coupon - $100` over time (2021-2022)

# COMMAND ----------

amount_over_time_coupon = df_reomm_p_cleaned_w_coupon[[
    "year",
    "month",
    "day",
    "AMOUNT"
]].groupby([
    "year",
    "month",
    "day",
]).sum().reset_index()
amount_over_time_coupon['date'] = pd.to_datetime(amount_over_time[['year', 'month', 'day']])

# COMMAND ----------

pivot_table_coupon = amount_over_time_coupon.pivot_table(index='month', columns='year', values='AMOUNT')
pivot_table_coupon.plot(kind='line')
plt.title('MoM trend on coupon usage (2021 & 2022)')
plt.show()

# COMMAND ----------

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data
ax.plot(amount_over_time_coupon['date'], amount_over_time_coupon['AMOUNT'])

# Calculate the moving averages
ma_30 = amount_over_time_coupon['AMOUNT'].rolling(window=30).mean()
ema_30 = amount_over_time_coupon['AMOUNT'].ewm(span=30, adjust=False).mean()

# Calculate the bollinger bands
ma_20 = amount_over_time_coupon['AMOUNT'].rolling(window=20).mean()
std_20 = amount_over_time_coupon['AMOUNT'].rolling(window=20).std()
upper_band = ma_20 + (2 * std_20)
lower_band = ma_20 - (2 * std_20)

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data and moving averages
ax.plot(amount_over_time_coupon['date'], amount_over_time_coupon['AMOUNT'], label='Amount')
ax.plot(amount_over_time_coupon['date'], ma_30, label='MA(30)')
ax.plot(amount_over_time_coupon['date'], ema_30, label='EMA(30)')

# Plot the bollinger bands
ax.plot(amount_over_time_coupon['date'], upper_band, label='Upper Band')
ax.plot(amount_over_time_coupon['date'], lower_band, label='Lower Band')

# Add a legend
ax.legend()


# Set the title and axis labels
ax.set_title('Coupon usage amount over Time (2021-2022)')
ax.set_xlabel('Date')
ax.set_ylabel('Amount')

# Show the plot
plt.show()

# COMMAND ----------

amount_over_time.set_index('date', inplace=True)

# COMMAND ----------

# import statsmodels.api as sm

# # Create the ARIMA model
# model = sm.tsa.ARIMA(amount_over_time['AMOUNT'], order=(1,1,1))

# # Fit the model
# results = model.fit()

# # Print the summary of the model
# print(results.summary())

# # Plot the residuals
# fig, ax = plt.subplots()
# ax.plot(results.resid)
# ax.set_title('Residuals of ARIMA Model')
# ax.set_xlabel('Date')
# ax.set_ylabel('Residual')
# plt.show()

# COMMAND ----------

# Split the dataset into train and test sets
train_size = int(len(amount_over_time) * 0.8)
train, test = amount_over_time[:train_size], amount_over_time[train_size:]

# Define the orders for the ARIMA models
orders = [
        (1, 1, 0), 
        (1, 1, 1), 
        (2, 1, 0), 
        (2, 1, 1)
    ]

# Train and evaluate each model
for order in orders:
    # Create the ARIMA model
    model = sm.tsa.ARIMA(train['AMOUNT'], order=order)

    # Fit the model
    results = model.fit()

    # Make predictions on the test set
    predictions = results.forecast(steps=len(test))[0]

    # Evaluate the model using the mean squared error (MSE)
    mse = mean_squared_error(test['AMOUNT'], predictions)

    # Print the order and the MSE
    print(f"ARIMA{order} MSE: {mse:.2f}")

    # Print the summary of the model
    print(results.summary())

    # Plot the residuals
    fig, ax = plt.subplots()
    ax.plot(results.resid)
    ax.set_title(f'Residuals of ARIMA{order} Model')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual')
    plt.show()

    # Plot the predictions and the actual values
    plt.plot(test.index, test['AMOUNT'], label='Actual')
    plt.plot(test.index, predictions, label='Predicted')
    plt.title(f"ARIMA{order} Predictions")
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.show()

# COMMAND ----------

# MAGIC %md #### grid search for ARIMA models

# COMMAND ----------

# Define the range of orders to search over
p = range(0, 3)
d = range(0, 2)
q = range(0, 3)
orders = list(itertools.product(p, d, q))

# Fit and evaluate each model
best_order = None
best_mse = float('inf')

for order in orders:
    try:
        print(f"running ARIMA{order} ... ")
        # Fit the ARIMA model
        model = sm.tsa.ARIMA(train['AMOUNT'], order=order)
        model_fit = model.fit()
        
        # Make predictions on the test set
        predictions = model_fit.forecast(steps=len(test))[0]
        
        # Evaluate the model using mean squared error (MSE)
        mse = mean_squared_error(test['AMOUNT'], predictions)
        
        # Check if this is the best model so far
        if mse < best_mse:
            best_mse = mse
            best_order = order
            
    except:
        continue
        
print(f"Best order: {best_order}, Best MSE: {best_mse:.2f}")

# COMMAND ----------

# MAGIC %md #### ARIMA with best order (1, 1, 2) prediction and evaluation

# COMMAND ----------

best_order

# COMMAND ----------

# Create the ARIMA model
model = sm.tsa.ARIMA(amount_over_time['AMOUNT'], order=best_order)

# Fit the model
results = model.fit()

# Make predictions on the test set
predictions = results.forecast(steps=len(test))[0]

# Evaluate the model using the mean squared error (MSE)
mse = mean_squared_error(test['AMOUNT'], predictions)

# Print the order and the MSE
print(f"ARIMA{order} MSE: {mse:.2f}")

# Print the summary of the model
print(results.summary())

# Plot the residuals
fig, ax = plt.subplots()
ax.plot(results.resid)
ax.set_title(f'Residuals of ARIMA{best_order} Model')
ax.set_xlabel('Date')
ax.set_ylabel('Residual')
plt.show()

# Plot the predictions and the actual values
plt.plot(test.index, test['AMOUNT'], label='Actual')
plt.plot(test.index, predictions, label='Predicted')
plt.title(f"ARIMA{best_order} Predictions")
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md # LSTM dev

# COMMAND ----------

amount_over_time['date'] = pd.to_datetime(amount_over_time[['year', 'month', 'day']])
df_lstm = amount_over_time[["date", "AMOUNT"]]

# COMMAND ----------

df_lstm.head()

# COMMAND ----------

# add 1 feature: get the difference in sales compared to the previous month 
df_lstm_diff = df_lstm.copy()

#add previous sales to the next row
df_lstm_diff['prev_AMOUNT'] = df_lstm_diff['AMOUNT'].shift(1)

#drop the null values and calculate the difference
df_lstm_diff = df_lstm_diff.dropna()
df_lstm_diff['diff'] = (df_lstm_diff['AMOUNT'] - df_lstm_diff['prev_AMOUNT'])



# COMMAND ----------

df_lstm_diff.head()

# COMMAND ----------

# MAGIC %md #### quite stationary on AMOUNT and also diff_AMOUNT as well

# COMMAND ----------

plt.plot(df_lstm_diff['date'], df_lstm_diff['diff'])
plt.xlabel('Date')
plt.ylabel('Diff')
plt.title('Plot of Diff over Time')
plt.show()

# COMMAND ----------

# #create dataframe for transformation from time series to supervised
df_lstm_supervised = df_lstm_diff.drop(['prev_AMOUNT'],axis=1)

# COMMAND ----------

# MAGIC %md #### add 90 day lag features

# COMMAND ----------

#adding lags
for inc in range(1,90):
    field_name = 'lag_' + str(inc)
    df_lstm_supervised[field_name] = df_lstm_supervised['diff'].shift(inc)

#drop null values
df_lstm_supervised = df_lstm_supervised.dropna().reset_index(drop=True)

# COMMAND ----------

df_lstm_supervised.head(10)

# COMMAND ----------

# MAGIC %md #### evaluate how good the added lag_n features explain the variation in our label using Adjusted R-squared 

# COMMAND ----------

# MAGIC %md ###### single lag_n testing

# COMMAND ----------

# Import statsmodels.formula.api
import statsmodels.formula.api as smf# Define the regression formula
model = smf.ols(formula='diff ~ lag_1', data=df_lstm_supervised)# Fit the regression
model_fit = model.fit()# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

# COMMAND ----------

# MAGIC %md ####lag_1 explains 0.012% of the variation

# COMMAND ----------

# Define the regression formula
formula = 'diff ~ ' + ' + '.join(['lag_' + str(i) for i in range(1, 90)])
model = smf.ols(formula=formula, data=df_lstm_supervised)

# Fit the regression
model_fit = model.fit()

# Extract the adjusted R-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


# COMMAND ----------

# MAGIC %md ####lag_1~lag_89 explains 59.34% of the variation 
# MAGIC - ref: https://towardsdatascience.com/predicting-sales-611cb5a252de

# COMMAND ----------

# df_lstm_supervised.tail()

# COMMAND ----------

#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_lstm_model = df_lstm_supervised.drop(['AMOUNT','date'],axis=1)#split train and test set
predict_test_size = 90
train_set, test_set = df_lstm_model[0:-predict_test_size].values, df_lstm_model[-predict_test_size:].values

# COMMAND ----------



# COMMAND ----------

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

# COMMAND ----------

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# COMMAND ----------

# !pip install tensorflow==2.10.1

# COMMAND ----------

# !pip install -U tensorflow

# COMMAND ----------

# !pip install --user --ignore-installed --upgrade tensorflow 

# COMMAND ----------

#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# COMMAND ----------

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False)


# COMMAND ----------

# Plot the training loss
plt.plot(history.history['loss'])
# plt.plot(validation_data)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# COMMAND ----------

# Print the model summary
model.summary()

# COMMAND ----------

X_test.shape

# COMMAND ----------

X_test

# COMMAND ----------

y_pred = model.predict(X_test,batch_size=1) 

# COMMAND ----------

# MAGIC %md #### inverse transformation for scaling

# COMMAND ----------

#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    #print(np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

# COMMAND ----------

# df_predict_next_90 = df_lstm_supervised.copy()
# df_predict_next_90 = df_predict_next_90.drop(['AMOUNT','date'],axis=1)
# horizon = 90
# input_set = df_predict_next_90[90:].values
# scaler_new = MinMaxScaler(feature_range=(-1, 1))
# scaler_new = scaler_new.fit(input_set)
# input_set = input_set.reshape(input_set.shape[0], input_set.shape[1])
# input_set_scaled = scaler.transform(input_set)

# X_input_test = input_set_scaled[:, 1:]
# X_input_test = X_input_test.reshape(X_input_test.shape[0], 1, X_input_test.shape[1])

# # Predict the next 90 data points
# y_pred = model.predict(X_input_test, batch_size=1)

# # Reshape y_pred
# y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

# # Rebuild test set for inverse transform
# pred_test_set = []
# for index in range(0,horizon):
#     pred_test_set.append(np.concatenate([y_pred[index],X_input_test[index]],axis=1))

# # Reshape pred_test_set
# pred_test_set = np.array(pred_test_set)
# pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

# # len(pred_test_set)
# pred_test_set_inverted = scaler_new.inverse_transform(pred_test_set)


# start_date = '2023-01-01'
# num_days = 90

# new_date_range = pd.date_range(start=start_date, periods=num_days)


# #create dataframe that shows the predicted sales
# result_list = []
# sales_dates = list(new_date_range)
# act_AMOUNT = list(df_lstm[-91:].AMOUNT)
# for index in range(0,len(pred_test_set_inverted)):
#     result_dict = {}
#     result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_AMOUNT[index])
#     result_dict['date'] = sales_dates[index+1]
#     result_list.append(result_dict)
# df_result = pd.DataFrame(result_list)

# COMMAND ----------

# 

# COMMAND ----------

len(pred_test_set_inverted)

# COMMAND ----------

#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_lstm[-91:].date)
act_AMOUNT = list(df_lstm[-91:].AMOUNT)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_AMOUNT[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)

# COMMAND ----------

# MAGIC %md # Plot the actual and predicted AMOUNT

# COMMAND ----------



#merge with actual AMOUNT dataframe
df_AMOUNT_pred = pd.merge(df_lstm,df_result,on='date',how='left')#plot actual and predicted

# plt.figure(figsize=(12, 6))

# Plot actual and predicted values
plt.plot(df_AMOUNT_pred['date'], df_AMOUNT_pred['AMOUNT'], label='Actual')
plt.plot(df_AMOUNT_pred['date'], df_AMOUNT_pred['pred_value'], label='Predicted')
plt.title('AMOUNT Prediction')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.show()

# COMMAND ----------

#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred) * 2):
    #print(np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

# COMMAND ----------

df_AMOUNT_pred.tail()

# COMMAND ----------



# COMMAND ----------

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas import read_csv

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# load the data
# IMPORTANT: THE FILE household_power_consumption.csv MUST BE IMPORTED MANUALLY (go the README...)
# csv_directory = "../household_power_consumption.csv";
# dataset = read_csv(csv_directory)

# # Print the data
# print("======== Dataset Shape ========")
# print(dataset.shape)

# print("======== Dataset Head ========")
# print(dataset.head())


dataset = df_lstm

# This function returns past and future windows
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


# The first 1.500.000 lines of the dataset for train
# The rest 507.259 lines for validation
# train_split = 1500000

train_split = int(len(dataset) * 0.9)

# Setting seed to ensure reproducibility.
tf.random.set_seed(13)

# Forecast a univariate time series!
# Extract the "AMOUNT" from the dataset
univariate_dataset = dataset['AMOUNT']
univariate_dataset.index = dataset['date']
univariate_dataset.head()

# Print what extracted
print(univariate_dataset)

# Observe how this data looks across time
univariate_dataset.plot(subplots=True)

# Data normalization
univariate_dataset = univariate_dataset.values
tf.keras.utils.normalize(univariate_dataset)

# The model will be given the last 20 recorded Global_Active_Power observations,
# and needs to learn to predict the Global_Active_Power at the next time step.
univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(univariate_dataset,
                                           0,
                                           train_split,
                                           univariate_past_history,
                                           univariate_future_target)

x_val_uni, y_val_uni = univariate_data(univariate_dataset,
                                       train_split,
                                       None,
                                       univariate_past_history,
                                       univariate_future_target)

# Print what univariate_data calculated for training
print('Single window of past history')
print(x_train_uni[0])
print('\n Target AMOUNT to predict')
print(y_train_uni[0])


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt


sample_plot = show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
sample_plot.show()


# Baseline Prediction
def baseline(history):
    return np.mean(history)


baseline_plot = show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, 'Baseline Prediction Example')
baseline_plot.show()

# Recurrent Neural Network - LSTM

batch_size = 5000
buffer_size = 100000

# Batch the data, put them into buffers, shuffle and cache them
train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.take(batch_size).shuffle(buffer_size).batch(batch_size).cache().repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.take(batch_size).batch(batch_size).cache().repeat()

# Create the LSTM model
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

# Compile the Model
lstm_model.compile(optimizer='adam', loss='mae')

for x, y in val_univariate.take(1):
    print(lstm_model.predict(x).shape)

# Train the Model
epochs = 10
steps_per_epoch = 200

trained_lstm = lstm_model.fit(
    train_univariate,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_univariate,
    validation_steps=50)

# Plot LSMT Model's 3 predictions
for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      lstm_model.predict(x)[0]], 0, 'LSTM model')
    plot.show()

# Plot loss
loss = trained_lstm.history['loss']
val_loss = trained_lstm.history['val_loss']
epochs_range = range(epochs)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# COMMAND ----------

# MAGIC %md #### forecast with prophet (future days: 365)

# COMMAND ----------

!python -m pip install prophet

# COMMAND ----------

from prophet import Prophet

# COMMAND ----------

df_prophet = df_lstm.copy()
df_prophet = df_prophet.rename(columns={'date': 'ds', 'AMOUNT':'y'})

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=365)
future.tail()

# COMMAND ----------

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# COMMAND ----------

fig = plt.figure(figsize=(15, 6))
fig1 = m.plot(forecast)
plt.title("2023 Forecast")
plt.show()

# COMMAND ----------

fig2 = m.plot_components(forecast)

# COMMAND ----------

# MAGIC %md #### plotly chart (same as above )

# COMMAND ----------

from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)

# COMMAND ----------

plot_components_plotly(m, forecast)

# COMMAND ----------

# MAGIC %md #### forecast with prophet (future days: 30)

# COMMAND ----------

df_prophet_30 = df_lstm.copy()
df_prophet_30 = df_prophet.rename(columns={'date': 'ds', 'AMOUNT':'y'})

m_30 = Prophet(yearly_seasonality=True, daily_seasonality=True)
m_30.fit(df_prophet_30)
future_30 = m_30.make_future_dataframe(periods=30)
# future_30.tail()

forecast_30 = m_30.predict(future_30)
# forecast_30[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

plot_plotly(m_30, forecast_30)

# COMMAND ----------

plot_components_plotly(m_30, forecast_30)

# COMMAND ----------


fig = m.plot(forecast)
plt.gca().set_title('2023 Forecast (30 future days)')
a = add_changepoints_to_plot(plt.gca(), m_30, forecast_30)
plt.show()


# COMMAND ----------

# MAGIC %md #### finetune prophet model
# MAGIC - ref: https://towardsdatascience.com/time-series-analysis-with-facebook-prophet-how-it-works-and-how-to-use-it-f15ecf2c0e3a

# COMMAND ----------

# instantiate the model and fit the timeseries
prophet = Prophet(weekly_seasonality=False, changepoint_range=1,changepoint_prior_scale=0.75)
prophet.fit(df_prophet_30)

# create a future data frame 
future_30_f = prophet.make_future_dataframe(periods=30)
forecast_30_f = prophet.predict(future_30_f)

# display the most critical output columns from the forecast
# forecast_30_f[['ds','yhat','yhat_lower','yhat_upper']].head()

# plot
# fig = prophet.plot(forecast_30_f)

# plot the forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast_30_f['ds'], forecast_30_f['yhat'], label='Forecast')
ax.fill_between(forecast_30_f['ds'], forecast_30_f['yhat_lower'], forecast_30_f['yhat_upper'], alpha=0.3, label='Confidence Interval')
ax.plot(df_prophet_30['ds'], df_prophet_30['y'], label='Actual')
ax.legend()
ax.set_title('2023 Forecast (30 future days)')
plt.show()


# COMMAND ----------

df_prophet_90 = df_lstm.copy()
df_prophet_90 = df_prophet.rename(columns={'date': 'ds', 'AMOUNT':'y'})

m_90 = Prophet(yearly_seasonality=True, daily_seasonality=True)
m_90.fit(df_prophet_90)
future_90 = m_90.make_future_dataframe(periods=90)
# future_30.tail()

forecast_90 = m_90.predict(future_90)
# forecast_30[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

plot_plotly(m_90, forecast_90)



# COMMAND ----------


# instantiate the model and fit the timeseries
prophet = Prophet(weekly_seasonality=False, changepoint_range=1,changepoint_prior_scale=0.75)
prophet.fit(df_prophet_90)

# create a future data frame 
future_90_f = prophet.make_future_dataframe(periods=90)
forecast_90_f = prophet.predict(future_90_f)

# display the most critical output columns from the forecast
# forecast_30_f[['ds','yhat','yhat_lower','yhat_upper']].head()

# plot
# fig = prophet.plot(forecast_30_f)

# plot the forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast_90_f['ds'], forecast_90_f['yhat'], label='Forecast')
ax.fill_between(forecast_90_f['ds'], forecast_90_f['yhat_lower'], forecast_90_f['yhat_upper'], alpha=0.3, label='Confidence Interval')
ax.plot(df_prophet_90['ds'], df_prophet_90['y'], label='Actual')
ax.legend()
ax.set_title('2023 Forecast (90 future days)')
plt.show()

# COMMAND ----------

