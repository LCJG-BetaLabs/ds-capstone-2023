# Databricks notebook source
pip install pycocotools

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import PIL.ImageFont as ImageFont
import copy
import json
import sys

from pycocotools import mask
from scipy.special import softmax

import os
import tensorflow.compat.v1 as tf


team_container = "capstone2023-hku-team-a"
team_path = f"abfss://{team_container}@capstone2023hku.dfs.core.windows.net/"
 
# Load the Fashionpedia Model
saved_model_dir = "/dbfs/fashionpedia_model"
dbutils.fs.cp(os.path.join(team_path, "fashionpedia_model"), saved_model_dir, recurse=True) # copy from ABFS to local
 
session = tf.Session(graph=tf.Graph())
_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)

# COMMAND ----------

fashionpedia_dir = "/dbfs/fashionpedia_api"
dbutils.fs.cp(os.path.join(team_path, "fashionpedia_api"), fashionpedia_dir, recurse=True) # copy from ABFS to local
ontology = json.load(open(os.path.join(fashionpedia_dir, "data/demo/category_attributes_descriptions.json")))
n_class = len(ontology['categories'])
score_th = 0.05

max_boxes_to_draw = 10

# Import data from Cloud Storage
images_path = "lanecrawford_img"
dbutils.fs.cp(os.path.join(team_path, "lanecrawford_img"), images_path, recurse=True) # copy from ABFS to local
a = dbutils.fs.ls(images_path)
segmentation = [] # create a list of lists of target segment for every image
output_folder = "lanecrawford_segmented"
dbutils.fs.mkdirs("lanecrawford_segmented")

# Do the segmentation for every image in the "lanecrawford_img" folder
for i in a:
    segment = []
    i = i.name
    id = i[:6]
    segment.append(id)
    image_path = "/dbfs/" + images_path + "/" + i
    with open(image_path, 'rb') as f:
        np_image_string = np.array([f.read()])
    image_raw = Image.open(image_path)
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
    for j in range(num_detections):
        prob = softmax(attribute_logits[j,:])
        attributes.append([k for k in range(len(prob)) if prob[k] > score_th])
    

    tops_list = [0, 1, 2, 3, 4, 5, 10, 11] # upperbody garment, dress and jumpsuit
    n = 0
    for p in detection_classes:
        category_id = p
        if n < max_boxes_to_draw:
            if (category_id in tops_list) and (detection_scores[n] > 0.5): # set the score threshold as 70%
                category_name = ontology['categories'][detection_classes[n]]['name']
                segment.append(category_name)
                area = list(detection_boxes[n,])
                attributes_str = ", ".join([ontology['attributes'][attr]['name'] for attr in attributes[n]])
                segment.append(attributes_str)

                # crop the image as the segment
                image = Image.open(image_path)
                top, left, bottom, right = area
                segmented_image = image.crop((left, top, right, bottom))
                output_image_path = os.path.join('/dbfs', output_folder, i)
                segmented_image.save(output_image_path, transparent=True)
                image.close()

                break

            else:
                n = n + 1
                if n == max_boxes_to_draw:
                    segment.extend(['None', 'None'])



    segmentation.append(segment)


dbutils.fs.cp(output_folder, os.path.join(team_path, "lanecrawford_segmented"), recurse=True) # copy folder from local to ABFS

# save the segment info as a csv file
segmentation_df = pd.DataFrame(segmentation, columns=['ID', 'Category', 'Attributes'])
dbutils.fs.mkdirs("segmentation")
segmentation_df.to_csv("/dbfs/segmentation/segmentation.csv", index = False)
dbutils.fs.cp("segmentation", os.path.join(team_path, "segmentation"), recurse=True) # copy folder from local to ABFS

# COMMAND ----------

segmentation_df

# COMMAND ----------

# print out the segmentation of an example image as a reference
max_boxes_to_draw = 8

linewidth = 2
fontsize = 10
line_alpha = 0.8
mask_alpha = 0.5


image_path = "/dbfs/lanecrawford_img/BVQ852.jpg" # change the file name to process different images
id_ = image_path[-10:-4]
with open(image_path, 'rb') as f:
    np_image_string = np.array([f.read()])
image = Image.open(image_path)
width, height = image.size
image_raw = np.array(image.getdata()).reshape(height, width, 3).astype(np.uint8)


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
for j in range(num_detections):
    prob = softmax(attribute_logits[j,:])
    attributes.append([k for k in range(len(prob)) if prob[k] > score_th])


cm_subsection = np.linspace(0., 1., min(max_boxes_to_draw, len(detection_scores))) 
colors = [cm.jet(x) for x in cm_subsection]

plt.figure()
fig, ax = plt.subplots(1)

for i in range(len(detection_scores)-1, -1, -1):
    if i < max_boxes_to_draw:
        
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
        
        font = ImageFont.load_default()
        text_width, text_height = font.getsize(detections_str)
        props = dict(boxstyle='Round, pad=0.05', facecolor=colors[i], linewidth=0, alpha=mask_alpha)
        ax.text(left, bottom, detections_str, fontsize=fontsize, verticalalignment='top', bbox=props)
        print(display_str)
        

plt.imshow(image, interpolation='none')
plt.axis('off')

dbutils.fs.mkdirs("segment_presentation")
output_example_path = os.path.join('/dbfs', 'segment_presentation', f"{id_}_segmentation.jpg")
plt.savefig(output_example_path, dpi=1000, transparent=True, bbox_inches='tight', pad_inches=0.05)


dbutils.fs.cp('segment_presentation', os.path.join(team_path, "segment_presentation"), recurse=True) # copy folder from local to ABFS