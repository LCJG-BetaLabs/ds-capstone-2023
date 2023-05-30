# Databricks notebook source
import pandas as pd
import os
df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))

# COMMAND ----------

df = df_cleansed.loc[df_cleansed['class']!=7].iloc[:,1:]
df['class'] = df['class']-1
df.groupby("class").size()

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train = pd.DataFrame()
X_test = pd.DataFrame()
X_val = pd.DataFrame()
y_train = []
y_test = []
y_val = []

for i in df['class'].unique():
    df_single_class = df.loc[df['class']==i]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df_single_class.iloc[:,:-1], df_single_class["class"], test_size=0.1,random_state=33)
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train1, y_train1, test_size=0.1, random_state=42)

    X_train = X_train.append(X_train1.reset_index(drop=True))
    X_val = X_val.append(X_val1.reset_index(drop=True))
    X_test = X_test.append(X_test1.reset_index(drop=True))

    y_train = y_train + y_train1.tolist() 
    y_test = y_test + y_test1.tolist() 
    y_val = y_val + y_val1.tolist() 


y_train = pd.DataFrame(y_train, columns=['class'])
y_test = pd.DataFrame(y_test, columns=['class']) 
y_val = pd.DataFrame(y_val, columns=['class'])



# COMMAND ----------

df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
df_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
df_val = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)

df_train.shape, df_test.shape, df_val.shape

# COMMAND ----------

from sklearn.utils import resample

# method 1
def dataResample(df, resample_size, output_path, RANDOM_STATE=123):
    df_resampled_final = pd.DataFrame()
    for cls in df['class'].unique().tolist():
        df_class = df.loc[df["class"]==cls]
        if df_class.shape[0]>resample_size:
            df_resampled = df_class.sample(n=resample_size)
        else:
            df_resampled = resample(df_class, replace=True, n_samples=resample_size, random_state=RANDOM_STATE)
        
        df_resampled_final = df_resampled_final.append(df_resampled)
    
    df_resampled_final.to_csv(output_path, index=False)

    return df_resampled_final

# COMMAND ----------

# method 2
def dataResample2(df, resample_size, resample_ratio, output_path, RANDOM_STATE=123):
    df_resampled_final = pd.DataFrame()
    for cls in df['class'].unique().tolist():
        df_class = df.loc[df["class"]==cls]
        if df_class.shape[0]>resample_size:
            df_resampled = df_class
        else:
            df_resampled = resample(df_class, replace=True, n_samples=int(df_class.shape[0]*resample_ratio), random_state=RANDOM_STATE)
        
        df_resampled_final = df_resampled_final.append(df_resampled)
    
    df_resampled_final.to_csv(output_path, index=False)

    return df_resampled_final

# COMMAND ----------

# method 3 resample by specific number
def dataResample3(df, resample_size_Dict, output_path, RANDOM_STATE=123):
    df_resampled_final = pd.DataFrame()
    for key in resample_size_Dict:
        df_class = df.loc[df["class"]==key]

        if resample_size_Dict[key] == 0:
            df_resampled = df_class
        else:
            df_resampled = resample(df_class, replace=True, n_samples=resample_size_Dict[key], random_state=RANDOM_STATE) 
        
        df_resampled_final = df_resampled_final.append(df_resampled)
    if output_path:
        df_resampled_final.to_csv(output_path, index=False)
    return df_resampled_final

# COMMAND ----------

resample_dict = {0:0,1:60,2:110,3:0,4:50,5:100}
df_train_method3 = dataResample3(df_train, resample_dict, None, RANDOM_STATE=123)
df_train_method3.groupby("class").size().plot.bar()

# COMMAND ----------

df_train_method3.to_csv('/dbfs/final_train_data3.csv', index=False)
df_test.to_csv('/dbfs/final_test_data3.csv', index=False)
df_val.to_csv('/dbfs/final_val_data3.csv', index=False)

# COMMAND ----------

df_train_m2 = dataResample2(df_train, 100, 2.5, '/dbfs/final_train_data2.csv')
df_train_m4 = dataResample(df_train, 100, '/dbfs/final_train_data4.csv')

# COMMAND ----------

df_test.to_csv('/dbfs/final_test_data.csv', index=False) # test data should not be resampled
df_test.to_csv('/dbfs/final_test_data2.csv', index=False) # test data should not be resampled
df_test.to_csv('/dbfs/final_test_data4.csv', index=False) # test data should not be resampled

# COMMAND ----------

df_val.to_csv('/dbfs/final_val_data.csv', index=False) # test data should not be resampled
df_val.to_csv('/dbfs/final_val_data2.csv', index=False) # test data should not be resampled
df_val.to_csv('/dbfs/final_val_data4.csv', index=False) # test data should not be resampled