# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import dataframe_image as dfi  
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data

# COMMAND ----------

df_label = pd.read_excel('Label_Check.xlsx')
df_label = df_label.iloc[:,:9]
df_label = df_label[df_label['label']==1]
df_label = df_label.reset_index(drop=True)

# COMMAND ----------

lc_id_all = df_label.loc[:,'lc']
ff_id_all = df_label.loc[:,'ff']

# COMMAND ----------

df_lc = pd.read_csv('lane_crawford.csv')
df_ff = pd.read_csv('farfetch.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preprocessing

# COMMAND ----------

lc_txt = df_lc.loc[:,['atg_code','prod_desc_eng','brand_desc','color_desc','compost','long_desc','price']]
lc_txt.index = lc_txt['atg_code']
lc_txt = lc_txt.drop(lc_txt.columns[[0]], axis = 1)
lc_txt = lc_txt.rename(columns = {'price':'Price(HKD)',
                          'prod_desc_eng':'Product_Name',
                          'brand_desc':'Brand',
                          'color_desc':'Color',
                          'compost':'Compost',
                          'long_desc':'Description'})
lc_txt['Product_Name'] = lc_txt['Product_Name'].str.upper()
lc_txt['Brand'] = lc_txt['Brand'].str.upper()
lc_txt['Color'] = lc_txt['Color'].str.upper()
lc_txt_select = lc_txt.loc[lc_txt.index.isin(lc_id_all)]
lc_txt_select['Price(HKD)'] = lc_txt_select['Price(HKD)'].astype(int)
lc_txt_select_t = lc_txt_select.T

# COMMAND ----------

ff_txt = df_ff.loc[:,['farfetch_id', 'pro_desc_eng', 'brand_desc', 'color_desc', 'compost', 'long_desc','price']]
ff_txt = ff_txt.iloc[:-1,]
ff_txt['farfetch_id'] = ff_txt['farfetch_id'].astype(int)
ff_txt.index = ff_txt['farfetch_id']
ff_txt = ff_txt.drop(ff_txt.columns[[0]], axis = 1)
ff_txt = ff_txt.rename(columns = {'price':'Price(HKD)',
                          'pro_desc_eng':'Product_Name',
                          'brand_desc':'Brand',
                          'color_desc':'Color',
                          'compost':'Compost',
                          'long_desc':'Description'})
ff_txt['Price(HKD)'] = ff_txt['Price(HKD)'].str.replace('HKD','').astype(int)
ff_txt['Product_Name'] = ff_txt['Product_Name'].str.upper()
ff_txt['Brand'] = ff_txt['Brand'].str.upper()
ff_txt['Color'] = ff_txt['Color'].str.upper()
ff_txt_select = ff_txt.loc[ff_txt.index.isin(ff_id_all)]
ff_txt_select_t = ff_txt_select.T

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image Visuliation

# COMMAND ----------

def image_visualization(df):
    for i in range(len(df)):
        lc_id = df['lc'][i]
        ff_id = str(df['ff'][i])
        image_filename1 = './lanecrawford_img/' + lc_id + '.jpg'
        image_filename2 = './farfetch_img/' + ff_id + '.jpg'
        img1 = cv2.imread(image_filename1)[:,:,(2,1,0)] 
        img2 = cv2.imread(image_filename2)[:,:,(2,1,0)]

        plt.subplot(121)
        plt.imshow(img1)
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(img2)
        plt.axis('off')

        plt.subplots_adjust(wspace=1.5)
        
        image_name = './image_img/' + lc_id + '_' + ff_id + '.jpg'
        plt.savefig(image_name, dpi=500,bbox_inches="tight")

# COMMAND ----------

image_visualization(df_label)

# COMMAND ----------

# MAGIC %md
# MAGIC ### add score and metrics in the image

# COMMAND ----------

# calculate price difference
def price_diff_cal(data):
    price_diff = []
    for i in range(len(data)):
        lc_id = data.iloc[i,0]
        ff_id = data.iloc[i,1]
        lc_series = lc_txt_select_t.loc['Price(HKD)',lc_id]
        ff_series = ff_txt_select_t.loc['Price(HKD)',ff_id]
        price = int((int(ff_series)/int(lc_series) - 1)*100)
        price_diff.append(price)
    data['price_difference'] = price_diff

# COMMAND ----------

price_diff_cal(df_label)

# COMMAND ----------

# calculate score
def total_score(data, weights):
    feature_cols = ['image','prod','brand','color','comps','long']
    weighted_scores = []
    for i in range(len(data)):
        scores = [data.loc[i,col]for col in feature_cols]
        weighted_score = sum(np.multiply(weights, scores))
        weighted_scores.append(weighted_score)
    data['scores'] = weighted_scores

# COMMAND ----------

weights = [0.5825737040054755,
           0.10609536032693793,
           0.27538137888796965,
           0.003324546593477956,
           0.025969564112296853,
           0.006655446073842374]

# COMMAND ----------

total_score(df_label, weights)

# COMMAND ----------

# combine score and metrics
df_metrics = df_label.copy()
final_metrics = df_metrics.drop(df_metrics.columns[[8]], axis = 1)
final_metrics[['image', 'prod', 'brand', 'color', 'comps', 'long',
       'scores']] = final_metrics[['image', 'prod', 'brand', 'color', 'comps', 'long',
       'scores']].round(2)

# COMMAND ----------

final_metrics_t = final_metrics.T

# COMMAND ----------

def insert_metrics(data, metrics_t):
    for i in range(len(data)):
        # open image
        lc_id = metrics_t.loc[:,i][0]
        ff_id = str(metrics_t.loc[:,i][1])
        imageFile = "./image_img/" + lc_id +"_"+ff_id+".jpg"
        im1 = Image.open(imageFile)

        # create an object that can draw on the image
        draw = ImageDraw.Draw(im1)

        # set the font and font size to use
        font_path = 'Times New Roman Bold.ttf'
        font_size1 = 70
        font1 = ImageFont.truetype(font_path, font_size1)
        font_size2 = 50
        font2 = ImageFont.truetype(font_path, font_size2)

        # calculate the position of the first part of the text
        text_width1, text_height1 = draw.textsize(f"Score: {metrics_t.loc[:,i]['scores']}""\n", font=font1)
        x1 = (im1.width - text_width1) / 2
        y1 = ((im1.height - text_height1) / 2 - 200)

        # calculate the position of the second part of the text
        text_width2, text_height2 = draw.textsize(f"Metrics:""\n"
                            f"1. Diffrence in price point: {metrics_t.loc[:,i]['price_difference']}%""\n"
                            f"2. Image similarity: {metrics_t.loc[:,i]['image']}""\n"
                            f"3. Product name similarity: {metrics_t.loc[:,i]['prod']}""\n"
                            f"4. Brand similarity: {metrics_t.loc[:,i]['brand']}""\n"
                            f"5. Color similarity: {metrics_t.loc[:,i]['color']}""\n"
                            f"6. Compost similarity: {metrics_t.loc[:,i]['comps']}""\n"
                            f"7. Product description similarity: {metrics_t.loc[:,i]['long']}", font=font2)
        x2 = (im1.width - text_width2) / 2
        y2 = ((im1.height + text_height1) / 2 - 200)

        # draw two parts in the center of the image
        draw.text((x1, y1), f"Score: {metrics_t.loc[:,i]['scores']}""\n", fill="black", font=font1)
        draw.text((x2, y2), f"Metrics:""\n"
                            f"1. Diffrence in price point: {metrics_t.loc[:,i]['price_difference']}%""\n"
                            f"2. Image similarity: {metrics_t.loc[:,i]['image']}""\n"
                            f"3. Product name similarity: {metrics_t.loc[:,i]['prod']}""\n"
                            f"4. Brand similarity: {metrics_t.loc[:,i]['brand']}""\n"
                            f"5. Color similarity: {metrics_t.loc[:,i]['color']}""\n"
                            f"6. Compost similarity: {metrics_t.loc[:,i]['comps']}""\n"
                            f"7. Product description similarity: {metrics_t.loc[:,i]['long']}", fill="black", font=font2)

        draw = ImageDraw.Draw(im1)
        # save image
        im1.save("./image_info_img/" + lc_id + "_" + ff_id + ".jpg")

# COMMAND ----------

insert_metrics(df_label, final_metrics_t)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Visualization

# COMMAND ----------

def text_visualization(data, lc, ff):
    for i in range(len(lc_id_all)):
        lc_id = data.iloc[i,0]
        ff_id = data.iloc[i,1]
        lc_series = lc.loc[:,lc_id]
        ff_series = ff.loc[:,ff_id]
        df = pd.concat([lc_series,ff_series],axis=1,ignore_index=False)
        df = df.style.set_properties(**{"background":"white",  # background color 
                               "width":'300px','font-size':'10px','font-family': 'Times New Roman',
                               "color":"black",  # font color  
                               "border-color":"white"})  # border
        image_filename = './txt_info_img/' + lc_id + '|' + str(ff_id) + '.jpg'
        table_styles = [
            {
                'selector': 'th, td',
                'props': [
                    ('font-family', 'Times New Roman'),
                    ('font-size', '10px'),
                    ('color', 'black'),
                    ('text-align', 'left')
                ]
            }
        ]
        df = df.set_table_styles(table_styles)
        dfi.export(df, image_filename, max_rows=None, max_cols=None, table_conversion='chrome', chrome_path=None,dpi=500)

# COMMAND ----------

text_visualization(final_metrics, lc_txt_select_t, ff_txt_select_t)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combine Images in to ONE

# COMMAND ----------

def combine_one(data):
    for i in range(len(lc_id_all)):
        lc_id = data['lc'][i]
        ff_id = str(data['ff'][i])
        image_name = './image_info_img/' + lc_id + '_' + ff_id + '.jpg'
        image1 = Image.open(image_name)
        text_name = './txt_info_img/'+lc_id+'|'+str(ff_id)+'.jpg'
        image2 = Image.open(text_name)
        width1, height1 = image1.size
        width2, height2 = image2.size
        new_width = max(width1, width2)
        new_height = height1 + height2
        new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        new_image.paste(image1, (int((new_width - width1) / 2), 0))
        new_image.paste(image2, (int((new_width - width2) / 2), height1))
        filename = './matching_img/' + lc_id + '_' + ff_id + '.jpg'
        new_image.save(filename)

# COMMAND ----------

combine_one(df_label)