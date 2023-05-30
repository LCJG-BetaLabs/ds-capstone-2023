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

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 21, 9

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

# MAGIC %md #### `isbn_google_reomm_*.json` dataset (external)

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

df_isbn_google_reomm_cleaned.head()

# COMMAND ----------

# MAGIC %md #### enrich `RECOMMENDATION` dataset with external google dataset

# COMMAND ----------

 df_reomm_p_cleaned_enriched = df_reomm_p_cleaned.merge(df_isbn_google_reomm_cleaned, left_on='ISBN13', right_on='isbn', how='left')

# COMMAND ----------

df_reomm_p_cleaned_enriched['TITLE'] = df_reomm_p_cleaned_enriched['TITLE'].apply(lambda x:x.rstrip())

# COMMAND ----------

df_reomm_p_cleaned_enriched

# COMMAND ----------

# MAGIC %md #### Find distinct categories 

# COMMAND ----------

len(df_reomm_p_cleaned_enriched['categories'].unique())

# COMMAND ----------

# MAGIC %md #### find distinct SHOP_NO

# COMMAND ----------

len(df_reomm_p_cleaned_enriched['SHOP_NO'].unique())

# COMMAND ----------

# MAGIC %md #### find distinct authors

# COMMAND ----------

len(df_reomm_p_cleaned_enriched['authors'].unique())

# COMMAND ----------

# MAGIC %md #### find distinct publisher

# COMMAND ----------

len(df_reomm_p_cleaned_enriched['publisher'].unique())

# COMMAND ----------

df_reomm_p_cleaned_enriched

# COMMAND ----------

# exclude Group Cash Coupon - $100
df_reomm_p_cleaned_enriched = df_reomm_p_cleaned_enriched[df_reomm_p_cleaned_enriched['TITLE'] != "Group Cash Coupon - $100"]

# COMMAND ----------

# MAGIC %md # `SHOP_NO` level time series forecast

# COMMAND ----------

df_hts_shop = df_reomm_p_cleaned_enriched[[
    "AMOUNT",
    "SHOP_NO",
    "year",
    "month",
    "day"
]]

df_hts_shop["yyyy_mm_dd"] = df_hts_shop["year"] + "-" + df_hts_shop["month"] + "-" + df_hts_shop["day"]

df_hts_shop = df_hts_shop.drop(["year", "month", "day"], axis = 1)

# COMMAND ----------

df_hts_shop

# COMMAND ----------

df_hts_shop = df_hts_shop.groupby([
    "yyyy_mm_dd",
    "SHOP_NO"
]).sum().reset_index()

# COMMAND ----------

df_hts_shop

# COMMAND ----------

# MAGIC %md #### Hierarchical Forecast
# MAGIC - ref: https://github.com/Nixtla/hierarchicalforecast

# COMMAND ----------

!pip install hierarchicalforecast

# COMMAND ----------

# !pip install statsforecast datasetsforecast

# COMMAND ----------

Y_df = df_hts_shop.copy()
Y_df = Y_df.rename(columns={'yyyy_mm_dd': 'ds', 'AMOUNT':'y'})

# COMMAND ----------

Y_df

# COMMAND ----------

hiers = [
    ['SHOP_NO']
]

# COMMAND ----------

# 

# COMMAND ----------

from hierarchicalforecast.utils import aggregate

# COMMAND ----------

Y_df, S_df, tags = aggregate(Y_df, hiers)
Y_df['y'] = Y_df['y']/1e3
Y_df = Y_df.reset_index()

# COMMAND ----------

# Y_df

# COMMAND ----------

# train test split 
Y_test_df = Y_df.groupby('unique_id').tail(8)
Y_train_df = Y_df.drop(Y_test_df.index)

Y_test_df = Y_test_df.set_index('unique_id')
Y_train_df = Y_train_df.set_index('unique_id')


# COMMAND ----------

#Computing base forecasts
#The following cell computes the base forecasts for each time series in Y_df using the ETS model. Observe that Y_hat_df contains the forecasts but they are not coherent.


from statsforecast.models import ETS
from statsforecast.core import StatsForecast

fcst = StatsForecast(df=Y_train_df,
                     models=[ETS(season_length=4, model='ZMZ')], 
                     freq='QS', n_jobs=-1)
Y_hat_df = fcst.forecast(h=8, fitted=True)
Y_fitted_df = fcst.forecast_fitted_values()

# COMMAND ----------

# Reconcile forecasts
# The following cell makes the previous forecasts coherent using the HierarchicalReconciliation class. Since the hierarchy structure is not strict, we can't use methods such as TopDown or MiddleOut. In this example we use BottomUp and MinTrace.

from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.core import HierarchicalReconciliation

# COMMAND ----------

# The dataframe Y_rec_df contains the reconciled forecasts.
reconcilers = [
    BottomUp(),
    MinTrace(method='mint_shrink')
]
hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S=S_df, tags=tags)

# COMMAND ----------

Y_rec_df.head()

# COMMAND ----------

Y_rec_df

# COMMAND ----------

# Y_fitted_df

# COMMAND ----------

# Evaluation
# The HierarchicalForecast package includes the HierarchicalEvaluation class to evaluate the different hierarchies and also is capable of compute scaled metrics compared to a benchmark model.

# COMMAND ----------

from hierarchicalforecast.evaluation import HierarchicalEvaluation

# COMMAND ----------

def mase(y, y_hat, y_insample, seasonality=4):
    errors = np.mean(np.abs(y - y_hat), axis=1)
    scale = np.mean(np.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality]), axis=1)
    return np.mean(errors / scale)

eval_tags = {}
eval_tags['Total'] = tags['SHOP_NO']
eval_tags['All series'] = np.concatenate(list(tags.values()))

evaluator = HierarchicalEvaluation(evaluators=[mase])
evaluation = evaluator.evaluate(
    Y_hat_df=Y_rec_df, Y_test_df=Y_test_df,
    tags=eval_tags,
    Y_df=Y_train_df
)
evaluation = evaluation.reset_index().drop(columns='metric').drop(0).set_index('level')
evaluation.columns = ['Base', 'BottomUp', 'MinTrace(mint_shrink)']
evaluation.applymap('{:.2f}'.format)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

print(len(list(Y_df['unique_id'].unique())))

# COMMAND ----------

from statsforecast import StatsForecast

# COMMAND ----------

StatsForecast.plot(Y_df)

# COMMAND ----------

from statsforecast.models import (
    AutoARIMA,
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive
)


# Create a list of models and instantiation parameters
models = [
    AutoARIMA(season_length=24),
    HoltWinters(),
    Croston(),
    SeasonalNaive(season_length=24),
    HistoricAverage(),
    DOT(season_length=24)
]

# COMMAND ----------

# Instantiate StatsForecast class as sf
sf = StatsForecast(
    df=Y_df, 
    models=models,
    freq='H', 
    n_jobs=-1,
    fallback_model = SeasonalNaive(season_length=7)
)

# COMMAND ----------

forecasts_df = sf.forecast(h=48, level=[90])
forecasts_df.head()

# COMMAND ----------

sf.plot(Y_df,forecasts_df)

# COMMAND ----------



# COMMAND ----------

# Evaluate the model’s performance

from datasetsforecast.losses import mse, mae, rmse

def evaluate_cross_validation(df, metric):
    models = df.drop(columns=['ds', 'cutoff', 'y']).columns.tolist()
    evals = []
    for model in models:
        eval_ = df.groupby(['unique_id', 'cutoff']).apply(lambda x: metric(x['y'].values, x[model].values)).to_frame() # Calculate loss for every unique_id, model and cutoff.
        eval_.columns = [model]
        evals.append(eval_)
    evals = pd.concat(evals, axis=1)
    evals = evals.groupby(['unique_id']).mean(numeric_only=True) # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals['best_model'] = evals.idxmin(axis=1)
    return evals



# COMMAND ----------

evaluation_df = evaluate_cross_validation(crossvaldation_df, mse)

evaluation_df.head()

# COMMAND ----------

# Select the best model for every unique series (STORE_ID)

def get_best_model_forecast(forecasts_df, evaluation_df):
    df = forecasts_df.set_index('ds', append=True).stack().to_frame().reset_index(level=2) # Wide to long 
    df.columns = ['model', 'best_model_forecast'] 
    df = df.join(evaluation_df[['best_model']])
    df = df.query('model.str.replace("-lo-90|-hi-90", "", regex=True) == best_model').copy()
    df.loc[:, 'model'] = [model.replace(bm, 'best_model') for model, bm in zip(df['model'], df['best_model'])]
    df = df.drop(columns='best_model').set_index('model', append=True).unstack()
    df.columns = df.columns.droplevel()
    df = df.reset_index(level=1)
    return df

# COMMAND ----------

prod_forecasts_df = get_best_model_forecast(forecasts_df, evaluation_df)

prod_forecasts_df.head()

# COMMAND ----------

# https://nixtla.github.io/statsforecast/examples/getting_started_complete.html