# Databricks notebook source
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import Data

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1 Farfetch Data

# COMMAND ----------

# import farfetch data
team_container = "capstone2023-hku-team-a"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
team_acount_name = "capstone2023hku"
folder_path1 = "img_features"
folder_path2 = "nlp"
img_path1 = "FF_img_features.json"
txt_path1 = "farfetch_result_v2.json"

img_url1 = f"https://{team_acount_name}.blob.core.windows.net/{team_container}/{folder_path1}/{img_path1}?st=2023-03-09&se=2023-05-31&spr=https&sv=2021-06-08&si=hku-team-a&sr=c&sig=IAR5rxam3Qc502Z6Ar3p7m7pHPb4ptOp2MRa8rFFvNI%3D"
txt_url1 = f"https://{team_acount_name}.blob.core.windows.net/{team_container}/{folder_path2}/{txt_path1}?st=2023-03-09&se=2023-05-31&spr=https&sv=2021-06-08&si=hku-team-a&sr=c&sig=IAR5rxam3Qc502Z6Ar3p7m7pHPb4ptOp2MRa8rFFvNI%3D"

# Load data
df_ff_img = pd.read_json(img_url1)
df_ff_txt = pd.read_json(txt_url1)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2 Lane Crawford Data

# COMMAND ----------

img_path2 = "LC_img_features.json"
txt_path2 = "lc_result_v2.json"

img_url2 = f"https://{team_acount_name}.blob.core.windows.net/{team_container}/{folder_path1}/{img_path2}?st=2023-03-09&se=2023-05-31&spr=https&sv=2021-06-08&si=hku-team-a&sr=c&sig=IAR5rxam3Qc502Z6Ar3p7m7pHPb4ptOp2MRa8rFFvNI%3D"
txt_url2 = f"https://{team_acount_name}.blob.core.windows.net/{team_container}/{folder_path2}/{txt_path2}?st=2023-03-09&se=2023-05-31&spr=https&sv=2021-06-08&si=hku-team-a&sr=c&sig=IAR5rxam3Qc502Z6Ar3p7m7pHPb4ptOp2MRa8rFFvNI%3D"

# Load data
df_lc_img = pd.read_json(img_url2)
df_lc_txt = pd.read_json(txt_url2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cosine Similarity: Image Features

# COMMAND ----------

ff_img = df_ff_img.T
lc_img = df_lc_img.T
img_similarity_matrix = cosine_similarity(lc_img, ff_img)

# COMMAND ----------

df_similarity_img = pd.DataFrame(img_similarity_matrix, index = lc_img.index, columns = ff_img.index)
df_similarity_img

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Cosine Similarity: Text Features

# COMMAND ----------

# farfetch

# COMMAND ----------

data_ff_txt = df_ff_txt.iloc[:-1,]

# COMMAND ----------

data_ff_txt['farfetch_id'] = data_ff_txt['farfetch_id'].astype(int)
data_ff_txt.index = data_ff_txt['farfetch_id']

# COMMAND ----------

ff_text = data_ff_txt.drop(data_ff_txt.columns[[0]], axis = 1)
ff_txt = ff_text.sort_values(by='farfetch_id')

# COMMAND ----------

# LC

# COMMAND ----------

df_lc_txt.index = df_lc_txt['atg_code']
lc_text = df_lc_txt.drop(df_lc_txt.columns[[0]], axis = 1)
lc_txt = lc_text.sort_values(by='atg_code')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Product Description

# COMMAND ----------

ff_text_prod = ff_txt['prod_desc_vector']
ff_txt_prod = pd.DataFrame(ff_text_prod.values.tolist(), index=ff_text_prod.index)

# COMMAND ----------

lc_text_prod = lc_txt['prod_desc_vector']
lc_txt_prod = pd.DataFrame(lc_text_prod.values.tolist(), index=lc_text_prod.index)

# COMMAND ----------

ff_vector_prod = np.array(ff_txt_prod)
lc_vector_prod = np.array(lc_txt_prod)
similarity_matrix_txt_prod = cosine_similarity(lc_txt_prod,ff_txt_prod)

# COMMAND ----------

df_similarity_txt_prod = pd.DataFrame(similarity_matrix_txt_prod,index = lc_txt_prod.index, columns = ff_txt_prod.index)
df_similarity_txt_prod

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Brand

# COMMAND ----------

ff_text_brand = ff_txt['brand_vector']
ff_txt_brand = pd.DataFrame(ff_text_brand.values.tolist(), index=ff_text_brand.index)

# COMMAND ----------

lc_text_brand = lc_txt['brand_vector']
lc_txt_brand = pd.DataFrame(lc_text_brand.values.tolist(), index=lc_text_brand.index)

# COMMAND ----------

ff_vector_brand = np.array(ff_txt_brand)
lc_vector_brand = np.array(lc_txt_brand)
similarity_matrix_txt_brand = cosine_similarity(lc_txt_brand,ff_txt_brand)
df_similarity_txt_brand = pd.DataFrame(similarity_matrix_txt_brand,index = lc_txt_brand.index, columns = ff_txt_brand.index)
df_similarity_txt_brand

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Color

# COMMAND ----------

ff_text_color = ff_txt['color_vector']
ff_txt_color = pd.DataFrame(ff_text_color.values.tolist(), index=ff_text_color.index)

# COMMAND ----------

lc_text_color = lc_txt['color_vector']
lc_txt_color = pd.DataFrame(lc_text_color.values.tolist(), index=lc_text_color.index)

# COMMAND ----------

ff_vector_color = np.array(ff_txt_color)
lc_vector_color = np.array(lc_txt_color)
similarity_matrix_txt_color = cosine_similarity(lc_txt_color,ff_txt_color)
df_similarity_txt_color = pd.DataFrame(similarity_matrix_txt_color,index = lc_txt_color.index, columns = ff_txt_color.index)
df_similarity_txt_color

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Product Composition

# COMMAND ----------

ff_text_comps = ff_txt['compost_vector']
ff_txt_comps = pd.DataFrame(ff_text_comps.values.tolist(), index=ff_text_comps.index)

# COMMAND ----------

lc_text_comps = lc_txt['compost_vector']
lc_txt_comps = pd.DataFrame(lc_text_comps.values.tolist(), index=lc_text_comps.index)

# COMMAND ----------

ff_vector_comps = np.array(ff_txt_comps)
lc_vector_comps = np.array(lc_txt_comps)
similarity_matrix_txt_comps = cosine_similarity(lc_txt_comps,ff_txt_comps)
df_similarity_txt_comps = pd.DataFrame(similarity_matrix_txt_comps,index = lc_txt_comps.index, columns = ff_txt_comps.index)
df_similarity_txt_comps

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Long Description

# COMMAND ----------

ff_text_long = ff_txt['long_desc_vector']
ff_txt_long = pd.DataFrame(ff_text_long.values.tolist(), index=ff_text_long.index)

# COMMAND ----------

lc_text_long = lc_txt['long_desc_vector']
lc_txt_long = pd.DataFrame(lc_text_long.values.tolist(), index=lc_text_long.index)

# COMMAND ----------

df_txt_long = pd.concat([ff_txt_long, lc_txt_long],axis=0)

# COMMAND ----------

ff_vector_long = np.array(ff_txt_long)
lc_vector_long = np.array(lc_txt_long)
similarity_matrix_txt_long = cosine_similarity(lc_txt_long,ff_txt_long)
df_similarity_txt_long = pd.DataFrame(similarity_matrix_txt_long,index = lc_txt_long.index, columns = ff_txt_long.index)
df_similarity_txt_long

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cosine Similarity Set

# COMMAND ----------

lc_list = []
for i in range(len(lc_img.index)):
    for j in range(len(ff_img.index)):
        lc_list.append((lc_img.index[i]))
lc_list

# COMMAND ----------

ff_list = []
for i in range(len(lc_img.index)):
    for j in range(len(ff_img.index)):
        ff_list.append((ff_img.index[j]))
ff_list

# COMMAND ----------

image_list = []
for i in range(len(lc_img.index)):
    for j in range(len(ff_img.index)):
        image_list.append(df_similarity_img.iloc[i,j])
image_list

# COMMAND ----------

prod_list = []
for i in range(len(lc_img.index)):
    for j in range(len(ff_img.index)):
        prod_list.append(df_similarity_txt_prod.iloc[i,j])
prod_list

# COMMAND ----------

brand_list = []
for i in range(len(lc_img.index)):
    for j in range(len(ff_img.index)):
        brand_list.append(df_similarity_txt_brand.iloc[i,j])
brand_list

# COMMAND ----------

color_list = []
for i in range(len(lc_img.index)):
    for j in range(len(ff_img.index)):
        color_list.append(df_similarity_txt_color.iloc[i,j])
color_list

# COMMAND ----------

comps_list = []
for i in range(len(lc_img.index)):
    for j in range(len(ff_img.index)):
        comps_list.append(df_similarity_txt_comps.iloc[i,j])
comps_list

# COMMAND ----------

long_list = []
for i in range(len(lc_img.index)):
    for j in range(len(ff_img.index)):
        long_list.append(df_similarity_txt_long.iloc[i,j])
long_list

# COMMAND ----------

similarity_set=pd.DataFrame({'lc':lc_list,'ff':ff_list,'image':image_list,'prod':prod_list, 'brand':brand_list, 'color':color_list,'comps':comps_list, 'long':long_list})

# COMMAND ----------

similarity_set

# COMMAND ----------

dbutils.fs.mkdirs("Cosine_similarity")
similarity_set.to_csv("/dbfs/Cosine_similarity/Cosine_similarity_v2.csv", index = False)
dbutils.fs.cp("Cosine_similarity", os.path.join(team_path, "Cosine_similarity"), recurse=True)