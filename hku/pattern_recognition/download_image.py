# Databricks notebook source
# function to download image
import requests
from PIL import Image
import io
import os
from pyspark.sql.functions import *

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
    container = "data3"
    data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"
    attr = spark.read.option("header", "true").option("escape", '\n').format("csv").load(os.path.join(data_path,"pattern_recognition", "attribute.csv"))

    # use the atg_code in attribute to save product image
    save_path = '/pattern_recognition/original/image'  # path to image folder


# COMMAND ----------

attr.display()

# COMMAND ----------

#(raw[1]['atg_code'])
#(re.sub(r'[\\\|[|\]\"\s\']+',' ',raw[1]['atg_code'])).strip()

# COMMAND ----------

i = 0
data = []
import re
from pyspark.sql import Row
raw = attr.collect()
while i < len(raw) - 1:
    if ']' in raw[i]['img_list']:
        img_list = re.sub(r'[\\\|[|\]\"\']+','',raw[i]['img_list']).strip()
        temp = raw[i].asDict()
        temp['img_list'] = img_list
        row = Row(**temp)
        data.append(row)
        i = i + 1
    else:
        img_list = (re.sub(r'[\\\|[|\]\"\']+','',raw[i]['img_list'])+re.sub(r'[\\\|[|\]\"\']+',' ',raw[i+1]['atg_code'])).strip()
        temp = raw[i].asDict()
        temp['img_list'] = img_list
        row = Row(**temp)
        data.append(row)
        i=i+2


# COMMAND ----------

df = spark.createDataFrame(data)
df.display()

# COMMAND ----------

outputPath = "/Workspace/Repos/Team B/cleaned_data.parquet"
df.write.mode("overwrite").parquet(outputPath)

# COMMAND ----------

filePath = "/Workspace/Repos/Team B/cleaned_data.parquet"
df = spark.read.parquet(filePath)
display(df)


# COMMAND ----------

df_1 = df.withColumn("img_list_split",split(col('img_list'), '\s+'))
df_1.display()

# COMMAND ----------

df_2 = df_1.collect()
save_path = '/pattern_recognition/original/image'
for item in df_2:
    code = item['atg_code']
    save_image(code,save_path=save_path)

# COMMAND ----------

#%sh
#ls /pattern_recognition/original/image

# COMMAND ----------

path1 ='/pattern_recognition/original/image'
test_img = 0
for i in os.listdir(path1):
    test_img = test_img+1
print("test_img:" + str(test_img) + "张")

# COMMAND ----------

pip install opencv-python

# COMMAND ----------

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# COMMAND ----------

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        #print(img)
        print(array_of_img)


# COMMAND ----------

#shape_of_img = []
def read_img_shape(directory_name):
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        #shape_of_img.append(img.shape)
        #print(img)
        print(img.shape)

# COMMAND ----------

#image = cv2.imread("file:/databricks/driver/file:/tmp/test_model/BVE870_in_xl.jpg")
a = Image.open('/pattern_recognition/original/image/BVV103_in_xl.jpg')
plt.imshow(a)
#print(image)
#plt.imshow(image)

# COMMAND ----------

# MAGIC %pip install torch==1.5.0
# MAGIC %pip install torchvision==0.6.0

# COMMAND ----------

import torch
import torchvision
#With change on brightness, saturation and contrast.

transform_train = torchvision.transforms.Compose([
    #torchvision.transforms.RandomResizedCrop([256, 256]),
    #torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
     #                                        ratio=(3.0/4.0, 4.0/3.0)),
    #torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.8,    # change the brightness, contrast, and saturation first
                                       saturation=0.8),
    torchvision.transforms.ToTensor()])

image_data_2 = torchvision.datasets.ImageFolder(
  root = '/pattern_recognition/original', transform = transform_train)

# COMMAND ----------

#compute standardization and mean
image_data = torchvision.datasets.ImageFolder(
  root = '/pattern_recognition/original', transform = transform_train) 
batch_size=64
from torch.utils.data import DataLoader
image_data_loader = DataLoader(
  image_data, 
  batch_size=batch_size,  
  num_workers=1
)


# COMMAND ----------

image_data

# COMMAND ----------


#compute standardization and mean
def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std
  
mean, std = batch_mean_and_sd(image_data_loader)
print("mean and std: \n", mean, std)

# COMMAND ----------

normalize = torchvision.transforms.Normalize( mean, std) #Normalization

transform_train_normalize = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    #torchvision.transforms.RandomHorizontalFlip(),
    #torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             #ratio=(3.0/4.0, 4.0/3.0)),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    torchvision.transforms.ToTensor(),
    normalize])   
#retransform image after Normalization
image_data_2 = torchvision.datasets.ImageFolder(
  root = '/pattern_recognition/original', transform = transform_train_normalize)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
imshow(image_data_2[27][0])

# COMMAND ----------

import os

# get all the image's name
def getAllFiles(targetDir):
    listFiles = os.listdir(targetDir)
    return listFiles

name = getAllFiles(r"/pattern_recognition/original/image")
name.sort()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import tensorflow as tf
import os
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    after_img = tf.clip_by_value(img, 0.0, 1.0)
    npimg = after_img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imsave(path,img):
  #  img = img / 2 + 0.5     # unnormalize
    after_img = tf.clip_by_value(img, 0.0, 1.0)
    npimg = after_img.numpy()
    plt.imsave(path,np.transpose(npimg, (1, 2, 0)))
#os.mkdir('/pattern_recognition/image_trans')
for i in range(len(image_data_2)):       #save all the images after preprocess
    imsave('/pattern_recognition/image_trans/'+name[i],image_data_2[i][0])




# COMMAND ----------



# COMMAND ----------

# MAGIC %sh
# MAGIC ls /pattern_recognition/image_trans

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /pattern_recognition

# COMMAND ----------

import os
path2 ='/pattern_recognition/image_trans'
tran_img = 0
for i in os.listdir(path2):
    tran_img = tran_img+1
print("tran_img:" + str(tran_img) + "张")

# COMMAND ----------

# MAGIC %pip install pycocotools
# MAGIC %pip install tensorflow

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import PIL.ImageFont as ImageFont
import copy
import json
import sys

from pycocotools import mask
from scipy.special import softmax
import tensorflow.compat.v1 as tf

# COMMAND ----------

import os
import tensorflow.compat.v1 as tf
 
#team_container = "capstone2023-hku-team-b"
#team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
 
saved_model_dir = "/dbfs/model"
#saved_model_dir = "/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/model"
dbutils.fs.cp(os.path.join("dbfs:/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com", "model"),saved_model_dir, recurse=True) # copy from ABFS to local
 
session = tf.Session(graph=tf.Graph())
_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)
 

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir /dbfs/image_trans_crop

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/image_trans_crop

# COMMAND ----------

name[27]

# COMMAND ----------

#crop the images
for i in range (len(name)):
    img = Image.open("/pattern_recognition/image_trans/"+name[i])
    cropped = img.crop((50, 20, 210, 200))  # (left, upper, right, lower)
    cropped.save("/dbfs/image_trans_crop/"+name[i])
    

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt
a = Image.open('/pattern_recognition/image_trans/BVG149_in_xl.jpg')
plt.imshow(a)

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt
a = Image.open('/dbfs/lanecrawford_img/BVG149.jpg')
plt.imshow(a)

# COMMAND ----------

a = Image.open('/dbfs/lanecrawford_prep_updated/BVG149.jpg')
plt.imshow(a)

# COMMAND ----------

img = Image.open("/dbfs/image_trans_crop/BVG149_in_xl.jpg")
plt.imshow(img)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/lanecrawford_segmented

# COMMAND ----------

import json
from PIL import Image

#An example of image segmentation
ontology = json.load(open('/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/demo/category_attributes_descriptions.json'))
n_class = len(ontology['categories'])
score_th = 0.05

image_path = '/dbfs/image_trans_crop/BVG115_in_xl.jpg'
#'/pattern_recognition/image_trans/BVO574_in_xl.jpg'
#'/pattern_recognition/image_trans/1.png'
with open(image_path, 'rb') as f:
    np_image_string = np.array([f.read()])
image_raw = Image.open(image_path).convert('RGB')
width, height = image_raw.size
image_raw = np.array(image_raw.getdata()).reshape(height, width, 3).astype(np.uint8)
plt.imshow(image_raw, interpolation='none')
plt.axis('off')


num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, detection_logits, attribute_logits, image_info = session.run(
    ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0', 'DetectionLogits:0', 'AttributeLogits:0', 'ImageInfo:0'],
    feed_dict={'Placeholder:0': np_image_string})
num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
detection_boxes = np.squeeze(detection_boxes * image_info[0, 2], axis=(0,))[0:num_detections]
detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections]
detection_masks = np.squeeze(detection_masks, axis=(0,))[0:num_detections]
detection_logits = np.squeeze(detection_logits, axis=(0,))[0:num_detections]
attribute_logits = np.squeeze(attribute_logits, axis=(0,))[0:num_detections]

# include attributes
attributes = []
for i in range(num_detections):
    prob = softmax(attribute_logits[i,:])
    attributes.append([j for j in range(len(prob)) if prob[j] > score_th])


# COMMAND ----------

#An example of image segmentation

max_boxes_to_draw = 5

linewidth = 2
fontsize = 10
line_alpha = 0.8
mask_alpha = 0.5

#output_image_path = '/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/demo/results.pdf'

image = copy.deepcopy(image_raw)

cm_subsection = np.linspace(0., 1., 5)
#cm_subsection = np.linspace(0., 1., min(max_boxes_to_draw, len(detection_scores))) 
colors = [cm.jet(x) for x in cm_subsection]

plt.figure()
fig, ax = plt.subplots(1)





for i in range(len(detection_scores)-1, -1, -1):
    if i < max_boxes_to_draw:
        # draw segmentation mask
        seg_mask = detection_masks[i,:,:]
        color = list(np.array(colors[i][:3])*255)
        pil_image = Image.fromarray(image)
        solid_color = np.expand_dims(
          np.ones_like(seg_mask), axis=2) * np.reshape(color, [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0*mask_alpha*seg_mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        image = np.array(pil_image.convert('RGB')).astype(np.uint8)
        
        # draw bbox
        top, left, bottom, right = detection_boxes[i,:]
        width = right - left
        height = bottom - top
        bbox = patches.Rectangle((left, top), width, height, 
                                 linewidth=linewidth, edgecolor=colors[i], 
                                 facecolor='none', alpha=line_alpha)
        ax.add_patch(bbox)
        # draw text
        attributes_str = ", ".join([ontology['attributes'][attr]['name'] for attr in attributes[i]])
        detections_str = '{} ({}%)'.format(ontology['categories'][detection_classes[i]-1]['name'],
                                           int(100*detection_scores[i]))
        display_str = '{}: {}'.format(detections_str, attributes_str)
        import matplotlib.font_manager as fm # to create font
        from PIL import Image,ImageFont,ImageDraw

        fontsize = 10
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)

        #font = ImageFont.truetype('arial.ttf', fontsize)

        text_width, text_height = font.getsize(detections_str)
        props = dict(boxstyle='Round, pad=0.05', facecolor=colors[i], linewidth=0, alpha=mask_alpha)
        ax.text(left, bottom, detections_str, fontsize=fontsize, verticalalignment='top', bbox=props)
        print(display_str)
        
plt.imshow(image, interpolation='none')
plt.axis('off')
#plt.savefig('/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/demo/result.pdf', transparent=True, bbox_inches='tight', pad_inches=0.05)

#tributes_str = ", ".join([ontology['attributes'][attr]['name'] for attr in attributes[i]])


# COMMAND ----------

len(ontology['categories'])

# COMMAND ----------

import json
from PIL import Image

#implement segmention on all images

ontology = json.load(open('/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/demo/category_attributes_descriptions.json'))
n_class = len(ontology['categories'])
score_th = 0.05

code=[]
category=[]
attribute=[]

for x in range (len(name)):
    image_path = '/pattern_recognition/image_trans_crop/'+name[x]
    with open(image_path, 'rb') as f:
        np_image_string = np.array([f.read()])
    image_raw = Image.open(image_path).convert('RGB')
    width, height = image_raw.size
    image_raw = np.array(image_raw.getdata()).reshape(height, width, 3).astype(np.uint8)


    num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, detection_logits, attribute_logits, image_info = session.run(
        ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0', 'DetectionLogits:0', 'AttributeLogits:0', 'ImageInfo:0'],
        feed_dict={'Placeholder:0': np_image_string})
    num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
    detection_boxes = np.squeeze(detection_boxes * image_info[0, 2], axis=(0,))[0:num_detections]
    detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
    detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections]
    detection_masks = np.squeeze(detection_masks, axis=(0,))[0:num_detections]
    detection_logits = np.squeeze(detection_logits, axis=(0,))[0:num_detections]
    attribute_logits = np.squeeze(attribute_logits, axis=(0,))[0:num_detections]

    # include attributes
    attributes = []
    for i in range(num_detections):
        prob = softmax(attribute_logits[i,:])
        attributes.append([j for j in range(len(prob)) if prob[j] > score_th])



    max_boxes_to_draw = 10

    linewidth = 2
    fontsize = 10
    line_alpha = 0.8
    mask_alpha = 0.5

    #output_image_path = '/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/demo/results.pdf'

    image = copy.deepcopy(image_raw)

    cm_subsection = np.linspace(0., 1., 10)
    #cm_subsection = np.linspace(0., 1., min(max_boxes_to_draw, len(detection_scores))) 
    colors = [cm.jet(x) for x in cm_subsection]

    for i in range(len(detection_scores)-1, -1, -1):
        if i < max_boxes_to_draw:
            # draw segmentation mask
            seg_mask = detection_masks[i,:,:]
            color = list(np.array(colors[i][:3])*255)
            pil_image = Image.fromarray(image)
            solid_color = np.expand_dims(
            np.ones_like(seg_mask), axis=2) * np.reshape(color, [1, 1, 3])
            pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
            pil_mask = Image.fromarray(np.uint8(255.0*mask_alpha*seg_mask)).convert('L')
            pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
            image = np.array(pil_image.convert('RGB')).astype(np.uint8)
            
            
            # draw text
            attributes_str = ", ".join([ontology['attributes'][attr]['name'] for attr in attributes[i]])
            detections_str = '{} ({}%)'.format(ontology['categories'][detection_classes[i]-1]['name'],
                                            int(100*detection_scores[i]))
            #display_str = '{}: {}'.format(detections_str, attributes_str)

            code.append(name[x])
            category.append(detections_str)
            attribute.append(attributes_str)


# COMMAND ----------

detection_classes[i]

# COMMAND ----------

code

# COMMAND ----------

import pandas as pd
for i in range(len(code)):
    code[i]=code[i].replace('_in_xl.jpg','')
df=pd.DataFrame(code, columns = ['code'])      #Create the dataframe df
df['category']=category
df['attributes']=attribute

# COMMAND ----------

df.display()

# COMMAND ----------

df.to_csv('segmentation.csv', index=False)

# COMMAND ----------

### 以下代码暂时无用

# COMMAND ----------

display_str

# COMMAND ----------

ontology['attributes'][attributes[1][0]]['name']

# COMMAND ----------

attributes_str = ", ".join([ontology['attributes'][attr]['name'] for attr in attributes[i]])
attributes_str

# COMMAND ----------

# MAGIC %pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# MAGIC %pip install fashionpedia

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC
# MAGIC import numpy as np
# MAGIC import os
# MAGIC
# MAGIC from fashionpedia.fp import Fashionpedia

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /pattern_recognition/image_trans

# COMMAND ----------

anno_file = "/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/sample.json"
img_root = "/pattern_recognition/image_trans"

# COMMAND ----------

# initialize Fashionpedia api
fp = Fashionpedia(anno_file)

# COMMAND ----------

cats = fp.loadCats(fp.getCatIds())
cat_names =[cat['name'] for cat in cats]
print('Fashionpedia categories: \n{}\n'.format('; '.join(cat_names)))

atts = fp.loadAttrs(fp.getAttIds())
att_names = [att["name"] for att in atts]
print('Fashionpedia attributes (all): \n{}\n'.format('; '.join(att_names)))

# COMMAND ----------



# COMMAND ----------

img = image_data_2[1][0] / 2 + 0.5
npimg = img.numpy()
plt.imsave('/pattern_recognition/image_trans/1.png',np.transpose(npimg, (1, 2, 0)))

# COMMAND ----------


after_img = tf.clip_by_value(image_data_2[1][0], 0.0, 1.0)

# COMMAND ----------

batch_size=64
iter_1 = torch.utils.data.DataLoader(image_data_2, batch_size, shuffle=False,
                                         drop_last=True)

# COMMAND ----------

iter_1

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
imshow(image_data_2[7][0])


# COMMAND ----------

# get some random training images
dataiter = iter(iter_1)
images, labels = dataiter.next()

# COMMAND ----------

imshow(image_data_2[7][0])

# COMMAND ----------

import torch
import torchvision
# show images
imshow(torchvision.utils.make_grid(images[1]))

# print labels
#print(' '.join('%5s' % image_data.classe

# COMMAND ----------

# MAGIC %pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# MAGIC %pip install fashionpedia

# COMMAND ----------

from fashionpedia.fp import Fashionpedia

# COMMAND ----------

import os


print(os.path.abspath(os.path.join('/pattern_recognition', "..")))


# COMMAND ----------

model = tf.keras.models.load_model("/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/model/saved_model.pb")

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC from matplotlib import cm
# MAGIC import matplotlib.patches as patches
# MAGIC import matplotlib.pyplot as plt
# MAGIC from PIL import Image
# MAGIC import numpy as np
# MAGIC import PIL.ImageFont as ImageFont
# MAGIC import copy
# MAGIC import json
# MAGIC import sys
# MAGIC
# MAGIC from pycocotools import mask
# MAGIC from scipy.special import softmax
# MAGIC import tensorflow.compat.v1 as tf

# COMMAND ----------

# =======================================================
#tf.reset_default_graph()

saved_model_dir = '/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/model'  # specify the model dir here
# =======================================================

session = tf.Session(graph=tf.Graph())

_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)

# COMMAND ----------

import json

ontology = json.load(open('/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/demo/category_attributes_descriptions.json'))
n_class = len(ontology['categories'])
score_th = 0.05

image_path = '/pattern_recognition/image_trans/1.png'
#'/pattern_recognition/image_trans/1.png'
with open(image_path, 'rb') as f:
    np_image_string = np.array([f.read()])
image_raw = Image.open(image_path).convert('RGB')
width, height = image_raw.size
image_raw = np.array(image_raw.getdata()).reshape(height, width, 3).astype(np.uint8)
plt.imshow(image_raw, interpolation='none')
plt.axis('off')


num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, detection_logits, attribute_logits, image_info = session.run(
    ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0', 'DetectionLogits:0', 'AttributeLogits:0', 'ImageInfo:0'],
    feed_dict={'Placeholder:0': np_image_string})
num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
detection_boxes = np.squeeze(detection_boxes * image_info[0, 2], axis=(0,))[0:num_detections]
detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections]
detection_masks = np.squeeze(detection_masks, axis=(0,))[0:num_detections]
detection_logits = np.squeeze(detection_logits, axis=(0,))[0:num_detections]
attribute_logits = np.squeeze(attribute_logits, axis=(0,))[0:num_detections]

# include attributes
attributes = []
for i in range(num_detections):
    prob = softmax(attribute_logits[i,:])
    attributes.append([j for j in range(len(prob)) if prob[j] > score_th])

# COMMAND ----------



# COMMAND ----------

/pattern_recognition/image_trans/'+str(i)+'.png'

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /pattern_recognition/image_trans

# COMMAND ----------

image_path = '/pattern_recognition/image_trans/100.png'
#'/pattern_recognition/image_trans/1.png'
with open(image_path, 'rb') as f:
    np_image_string = np.array([f.read()])
image_raw = Image.open(image_path).convert('RGB')
image_raw

# COMMAND ----------

import json

ontology = json.load(open('/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/demo/category_attributes_descriptions.json'))
n_class = len(ontology['categories'])
score_th = 0.05

image_path = '/pattern_recognition/image_trans/1.png'
#'/pattern_recognition/image_trans/1.png'
with open(image_path, 'rb') as f:
    np_image_string = np.array([f.read()])
image_raw = Image.open(image_path).convert('RGB')
width, height = image_raw.size
image_raw = np.array(image_raw.getdata()).reshape(height, width, 3).astype(np.uint8)
plt.imshow(image_raw, interpolation='none')
plt.axis('off')


num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, detection_logits, attribute_logits, image_info = session.run(
    ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0', 'DetectionLogits:0', 'AttributeLogits:0', 'ImageInfo:0'],
    feed_dict={'Placeholder:0': np_image_string})
num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
detection_boxes = np.squeeze(detection_boxes * image_info[0, 2], axis=(0,))[0:num_detections]
detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections]
detection_masks = np.squeeze(detection_masks, axis=(0,))[0:num_detections]
detection_logits = np.squeeze(detection_logits, axis=(0,))[0:num_detections]
attribute_logits = np.squeeze(attribute_logits, axis=(0,))[0:num_detections]

# include attributes
attributes = []
for i in range(num_detections):
    prob = softmax(attribute_logits[i,:])
    attributes.append([j for j in range(len(prob)) if prob[j] > score_th])

# COMMAND ----------

max_boxes_to_draw = 10

linewidth = 2
fontsize = 10
line_alpha = 0.8
mask_alpha = 0.5

output_image_path = '/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/demo/results.pdf'

image = copy.deepcopy(image_raw)

cm_subsection = np.linspace(0., 1., 10)
#cm_subsection = np.linspace(0., 1., min(max_boxes_to_draw, len(detection_scores))) 
colors = [cm.jet(x) for x in cm_subsection]

plt.figure()
fig, ax = plt.subplots(1)

for i in range(len(detection_scores)-1, -1, -1):
    if i < max_boxes_to_draw:
        # draw segmentation mask
        seg_mask = detection_masks[i,:,:]
        color = list(np.array(colors[i][:3])*255)
        pil_image = Image.fromarray(image)
        solid_color = np.expand_dims(
          np.ones_like(seg_mask), axis=2) * np.reshape(color, [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0*mask_alpha*seg_mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        image = np.array(pil_image.convert('RGB')).astype(np.uint8)
        
        # draw bbox
        top, left, bottom, right = detection_boxes[i,:]
        width = right - left
        height = bottom - top
        bbox = patches.Rectangle((left, top), width, height, 
                                 linewidth=linewidth, edgecolor=colors[i], 
                                 facecolor='none', alpha=line_alpha)
        ax.add_patch(bbox)
        # draw text
        attributes_str = ", ".join([ontology['attributes'][attr]['name'] for attr in attributes[i]])
        detections_str = '{} ({}%)'.format(ontology['categories'][detection_classes[i]]['name'],
                                           int(100*detection_scores[i]))
        display_str = '{}: {}'.format(detections_str, attributes_str)
        import matplotlib.font_manager as fm # to create font
        from PIL import Image,ImageFont,ImageDraw

        fontsize = 10
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)

        #font = ImageFont.truetype('arial.ttf', fontsize)

        text_width, text_height = font.getsize(detections_str)
        props = dict(boxstyle='Round, pad=0.05', facecolor=colors[i], linewidth=0, alpha=mask_alpha)
        ax.text(left, bottom, detections_str, fontsize=fontsize, verticalalignment='top', bbox=props)
        print(display_str)
        
plt.imshow(image, interpolation='none')
plt.axis('off')
plt.savefig('/dbfs/FileStore/shared_uploads/capstone2023_hku_team_b@ijam.onmicrosoft.com/data/demo/result.pdf', transparent=True, bbox_inches='tight', pad_inches=0.05)

# COMMAND ----------



# COMMAND ----------

