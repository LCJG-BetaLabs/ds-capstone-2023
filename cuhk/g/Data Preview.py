# Databricks notebook source
!pip install datasets

# COMMAND ----------

!pip install ml_things

# COMMAND ----------

import os
import pandas as pd
import ast
import pyspark.dbutils
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
from datasets import load_metric



# LC
container = "data3" 
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
df2 = spark.read.format("csv").load(os.path.join(data_path, "pattern_recognition", "attribute.csv"))
dbutils.fs.cp(os.path.join(data_path, "pattern_recognition"), "file:/pattern_recognition", recurse=True) # copy folder from ABFS to local

# COMMAND ----------

dbutils.fs.ls("file:/dbfs/Users/capstone2023_cuhk_team_g@ijam.onmicrosoft.com/")

# COMMAND ----------

os.listdir()

# COMMAND ----------

dbutils.fs.help()

# COMMAND ----------

pdf = df2.toPandas()

# COMMAND ----------

display(df2)

# COMMAND ----------

pdf.loc[pdf['_c13'].isna()]

# COMMAND ----------

new_pdf = pdf.copy()

# COMMAND ----------

pdf.columns = pdf.iloc[0]
pdf = pdf[1:]
pdf.head(10)

# COMMAND ----------

pdf.head(1000)

# COMMAND ----------

pdf.columns = [         'atg_code',     'prod_desc_eng',  'brand_group_desc',
              'brand_desc',       'atg_bu_desc',    'atg_class_desc',
       'atg_subclass_desc',        'color_desc',           'compost',
                    'care',        'size_n_fit',         'long_desc',
                   'price',          'img_list',            ]

# COMMAND ----------

suffix = pdf.loc[pdf['img_list'].isna()]
suffix['primaryKey'] = suffix['atg_code'].str.split('_').str[0].str.replace(" '", "")
suffix = suffix[['atg_code', 'primaryKey']]

# COMMAND ----------

df_cleansed = pdf.loc[pdf['img_list'].notna()]
df_cleansed = df_cleansed.merge(suffix, left_on='atg_code', right_on='primaryKey')
df_cleansed['img_list_combined'] = df_cleansed['img_list'] + df_cleansed['atg_code_y']

# COMMAND ----------

df_cleansed['img_list_combined'] = df_cleansed['img_list_combined'].str.replace(" ", ",")
df_cleansed['img_list_combined'] = df_cleansed['img_list_combined'].str.replace("\n", "")
df_cleansed['img_list_combined'] = df_cleansed['img_list_combined'].str.replace('"', '')

# COMMAND ----------

df_cleansed['care'] = df_cleansed['care'].str.replace("<li>","")
df_cleansed['size_n_fit'] = df_cleansed['size_n_fit'].str.replace("<li>","")
df_cleansed['care'] = df_cleansed['care'].str.replace("</li>","")
df_cleansed['size_n_fit'] = df_cleansed['size_n_fit'].str.replace("</li>","")

# COMMAND ----------

df_cleansed.head(10)

# COMMAND ----------

df_cleansed.shape[0]

# COMMAND ----------

res = ast.literal_eval(df_cleansed['img_list_combined'][0])

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df_cleansed['class'] = ""

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM `dbfs`;

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC SELECT * FROM `hive_metastore`.`default`.`items_pattern`;

# COMMAND ----------

class_df = _sqldf.toPandas()

# COMMAND ----------

#Check unique tags and their tally
class_df['tag'].value_counts()

# COMMAND ----------

df_cleansed = df_cleansed.drop(['atg_code_y','img_list','primaryKey'],axis=1)

# COMMAND ----------

df_cleansed.rename(columns={'atg_code_x':'atg_code'},inplace=True)

# COMMAND ----------

df_cleansed = df_cleansed.merge(class_df,how='left',on='atg_code')

# COMMAND ----------

df_cleansed['class'] = df_cleansed['tag']

# COMMAND ----------

del df_cleansed['tag']

# COMMAND ----------

def cat_to_numeric(x):
    if x == 'graphic_print': return 1
    if x == 'multi_color': return 2
    if x == 'word_print': return 3
    if x == 'plain': return 4
    if x == 'checks': return 5
    if x == 'stripe': return 6
    if x == 'untagged': return 7

# COMMAND ----------

df_cleansed['class'] = df_cleansed['class'].apply(cat_to_numeric)

# COMMAND ----------

df_cleansed.head(5)

# COMMAND ----------

df_cleansed['long_desc'].loc[1]

# COMMAND ----------

df_cleansed['long_desc']

# COMMAND ----------

###############GPT to do classification

# COMMAND ----------

dataset = Dataset.from_pandas(df_cleansed)

# COMMAND ----------

dataset = dataset.remove_columns(['atg_code','brand_group_desc','brand_desc','atg_bu_desc','atg_class_desc','size_n_fit', 'price','img_list_combined','__index_level_0__'])

# COMMAND ----------

dataset = dataset.shuffle(seed = 555)

# COMMAND ----------

train_testvalid = dataset.train_test_split(test_size=0.3)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})


# COMMAND ----------

model_name = "ydshieh/tiny-random-gptj-for-sequence-classification"

tokenizer = AutoTokenizer.from_pretrained(model_name, problem_type="multi_label_classification")

tokenizer.pad_token = tokenizer.eos_token

# COMMAND ----------

def tokenize_function(examples):
    return tokenizer(examples["long_desc"], truncation = True, max_length = 512, padding = True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# COMMAND ----------

tokenized_datasets = tokenized_datasets.rename_column("class", "labels")

# COMMAND ----------

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7, problem_type="multi_label_classification",ignore_mismatched_sizes=True)
# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# COMMAND ----------

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis = -1)
    return metric.compute(predictions=predictions, references=labels)

# COMMAND ----------

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size= 64,
                                  per_device_eval_batch_size= 64
                                  )

# COMMAND ----------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# COMMAND ----------

trainer.train()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

########################GPT method 2

# COMMAND ----------

import io
import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
 
# Set seed for reproducibility.
set_seed(123)
 
# Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
epochs = 4
 
# Number of batches - depending on the max sequence length and GPU memory.
# For 512 sequence length batch of 10 works without cuda memory issues.
# For small sequence length can try batch of 32 or higher.
batch_size = 32
 
# Pad or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = 60
 
# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# Name of transformers model - will use already pretrained model.
# Path of transformer model - will load your own model from local disk.
model_name_or_path = 'gpt2'
 
# Dictionary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids = {'graphic_print': 0, 'multi_color': 1,'word_print': 2, 'plain': 3,'checks': 4, 'stripe': 5, 'stripe': 6}
 
# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

# COMMAND ----------

class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
     
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.
 
    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.
 
    Arguments:
 
      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.
 
      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.
 
      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.
 
    """
 
    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
 
        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder
 
        return
 
    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.
 
        Arguments:
 
          item (:obj:`list`):
              List of texts and labels.
 
        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """
 
        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})
 
        return inputs

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

open("/dbfs/download_image.py", "r")

# COMMAND ----------

f = open("/dbfs/download_image.py", "r")
f

# COMMAND ----------

with open('/dbfs/download_image.py', encoding='utf8') as f:
    for line in f:
        print(line.strip())

# COMMAND ----------

# function to download image
import requests
from PIL import Image
import io
import os


def save_image(atg: str, save_path: str, postfix: str = "in_xl") -> None:
    """
    Saves product image given atg_code.
    Args:
    atg: product ID
    save_path: path to save the image
    """
    filename = os.path.join(save_path, f"{atg}_{postfix}.jpg")
    if not os.path.exists(filename):
        url = f"https://media.lanecrawford.com/{atg[0]}/{atg[1]}/{atg[2]}/{atg}_{postfix}.jpg"
        r = requests.get(url)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename)


if __name__ == '__main__':
    # load the attribute table
    attr = ...

    # use the atg_code in attribute to save product image
    save_path = ...  # path to image folder

# COMMAND ----------

atg = 'BVQ613'
postfix = 'in_xl'
url = f"https://media.lanecrawford.com/{atg[0]}/{atg[1]}/{atg[2]}/{atg}_{postfix}.jpg"
url

# COMMAND ----------

for atg in df_cleansed['atg_code'].tolist():
    print(atg)

# COMMAND ----------

for atg in df_cleansed['atg_code'].tolist():
    save_image(atg, '/dbfs/image/')

# COMMAND ----------

image_path_list = dbutils.fs.ls("file:/dbfs/image/")

# COMMAND ----------

display(dbutils.fs.ls("file:/dbfs/image/"))

# COMMAND ----------



# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def display_image(path, dpi=500):
    """
    Description:
        Displayes an image
    Inputs:
        path (str): File path
        dpi (int): Your monitor's pixel density
    """
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize = (width/dpi,height/dpi))
    plt.imshow(img, interpolation='nearest', aspect='auto')

# COMMAND ----------

for atg in df_cleansed['atg_code'].tolist()[:10]:
    display_image(f"/dbfs/image/{atg}_in_xl.jpg")

# COMMAND ----------



# COMMAND ----------

df_cleansed.to_csv('/dbfs/cleansed_data.csv')

# COMMAND ----------

dbutils.fs.put('file:/pattern_recognition/dataset/', df_cleansed)

# COMMAND ----------

team_container = "capstone2023-cuhk-team-b"
team_path = f"abfss://{team_container}@capstone2023cuhk.dfs.core.windows.net/"

# COMMAND ----------

sparkDF=spark.createDataFrame(df_cleansed)
sparkDF.write.mode('overwrite').parquet(os.path.join("file:/pattern_recognition/", "cleansed_data"))

# COMMAND ----------

df_cleansed

# COMMAND ----------

dbutils.fs.ls("file:/dbfs/")