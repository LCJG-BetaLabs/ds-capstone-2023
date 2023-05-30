# Databricks notebook source
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import re
import numpy as np
import pandas as pd

# COMMAND ----------

resnet152 = models.resnet152(pretrained=True)   # utilizing the available pretrained model
resnet152 = torch.nn.Sequential(*(list(resnet152.children())[:-1]))  # removing the last layer FC
resnet152.eval()

# COMMAND ----------

# resizing and centralizing

def preprocess_image(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = preprocess(img)
    return img_t.unsqueeze(0)

# COMMAND ----------

def extract_features(image_path, model):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = model(input_tensor)
    features = features.squeeze().numpy()
    return features

# COMMAND ----------

# MAGIC %md
# MAGIC #### Farfetch

# COMMAND ----------

# main function for FF

team_container = "capstone2023-hku-team-a"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"

images_path = "farfetch_img"
dbutils.fs.cp(os.path.join(team_path, "farfetch_img"), images_path, recurse=True) # copy from ABFS to local
pics = dbutils.fs.ls(images_path)  # return is a list

feature_list = []
id_list = []

# COMMAND ----------

# Calculating feature vectors
for i in pics:
    path = "/dbfs/" + images_path + "/" + i.name
    features = extract_features(path, resnet152)
    # save features
    feature_list.append(features)
    # save ids
    img_id = re.findall("(.*)\.jpg", i.name)[0]
    id_list.append(img_id)

# organizing to dataframe
result = pd.DataFrame({
    "id": id_list,
    "feature": feature_list
})
# But we found that using this dataframe to form a csv may cause some problems
# Directly saving this to csv file causes loss of data cuz it save the array as str

# COMMAND ----------

# One possible solution: transforming to str
# This may require some fussy steps when loading back to python(various transformation in data types)

result2 = result.copy()

# merge data as string
result2["feature"] = result2["feature"].map(lambda x: ','.join(map(str, x)))

# save the feature info to csv file
dbutils.fs.mkdirs("img_features")
result2.to_csv("/dbfs/img_features/FF_img_features.csv", index = False)
dbutils.fs.cp("img_features", os.path.join(team_path, "img_features"), recurse=True) # copy folder from local to ABFS

# COMMAND ----------

# Another possible solution: saving the feature info to json file
# This method is more convenient for loading data back in future
# The 1st keys are the item ids and 2nd keys are serials of features(0-2047)

dict1 = {}
for i in range(len(id_list)):
    dict1[id_list[i]] = feature_list[i]

result1 = pd.DataFrame(dict1)

dbutils.fs.mkdirs("img_features")
result1.to_json("/dbfs/img_features/FF_img_features.json")
dbutils.fs.cp("img_features", os.path.join(team_path, "img_features"), recurse=True) # copy folder from local to ABFS

# COMMAND ----------

# MAGIC %md
# MAGIC #### Lanecrawford

# COMMAND ----------

# main function for LC 

team_container = "capstone2023-hku-team-a"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"

images_path = "lanecrawford_segmented"
dbutils.fs.cp(os.path.join(team_path, "lanecrawford_segmented"), images_path, recurse=True) # copy from ABFS to local
pics = dbutils.fs.ls(images_path)

feature_list1 = []
id_list1 = []

# COMMAND ----------

for i in pics:
    path = "/dbfs/" + images_path + "/" + i.name
    features = extract_features(path, resnet152)
    # save features
    feature_list1.append(features)
    # save ids
    img_id = re.findall("(.*)\.jpg", i.name)[0]
    id_list1.append(img_id)

# COMMAND ----------

# save to csv

# organize to dataframe
result = pd.DataFrame({
    "id": id_list1,
    "feature": feature_list1
})

# merge data as string
result["feature"] = result["feature"].map(lambda x: ','.join(map(str, x)))

# save the feature info to csv file
dbutils.fs.mkdirs("img_features")
result.to_csv("/dbfs/img_features/LC_img_features.csv", index = False)
dbutils.fs.cp("img_features", os.path.join(team_path, "img_features"), recurse=True) # copy folder from local to ABFS

# COMMAND ----------

# save to json

dict2 = {}
for i in range(len(id_list1)):
    dict2[id_list1[i]] = feature_list1[i]

resultlc = pd.DataFrame(dict2)

dbutils.fs.mkdirs("img_features")
resultlc.to_json("/dbfs/img_features/LC_img_features.json")
dbutils.fs.cp("img_features", os.path.join(team_path, "img_features"), recurse=True) # copy folder from local to ABFS