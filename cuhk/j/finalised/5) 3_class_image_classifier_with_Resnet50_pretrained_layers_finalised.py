# Databricks notebook source
# MAGIC %md # This is the notebook that shows the steps taken to train a 3-class image classifier that :
# MAGIC   - takes in book cover thumbnail image for the respective 3 classes (high-, mid- and low-sales) labels
# MAGIC     - the 3 classes are split by 60th percentile and 90th percentile
# MAGIC       - reference notebook: https://adb-5911062106551859.19.azuredatabricks.net/?o=5911062106551859#notebook/1777744143329555/command/1777744143329556
# MAGIC         - relevant code: command 13 ~ 15

# COMMAND ----------

import matplotlib.pyplot as plotter_lib
import numpy as np
import PIL as image_lib
import tensorflow as tflow
from tensorflow.keras.layers import Flatten
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# COMMAND ----------

# MAGIC %md # define :
# MAGIC   - image size
# MAGIC   - path for the input images
# MAGIC   - train test split proportion

# COMMAND ----------

img_height,img_width=190,128
batch_size=32
train_ds = tflow.keras.preprocessing.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class_Subset/",
  validation_split=0.2,
  subset="training",
  seed=123,

label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(
  f"/dbfs/team_j_downloaded_images/image_Class_Subset/",
  validation_split=0.2,
  subset="validation",
  seed=123,
label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# COMMAND ----------

# MAGIC %md # define model layer and ResNet50 pretrained layer 

# COMMAND ----------

resnet_model = Sequential()

pretrained_model_for_demo= tflow.keras.applications.ResNet50(include_top=False,

                   input_shape=(190,128,3),

                   pooling='avg',classes=3,

                   weights='imagenet'
                   #weights= None
                   )

for each_layer in pretrained_model_for_demo.layers:
        each_layer.trainable=False

resnet_model.add(pretrained_model_for_demo)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
#resnet_model.add(Dense(256, activation='relu'))
#resnet_model.add(Dense(128, activation='relu'))
resnet_model.add(Dense(3, activation='softmax'))



# COMMAND ----------

# MAGIC %md # run model training with 10 epoch and 0.001 learning rate
# MAGIC   - accuracy: 0.9236 ; val_accuracy: 0.5222 ==> overfitting is encountered

# COMMAND ----------

resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

history = resnet_model.fit(train_ds, validation_data=validation_ds, epochs=10)



# COMMAND ----------

# MAGIC %md # plot for accuracy and loss on training and validation dataset

# COMMAND ----------

plotter_lib.figure(figsize=(8, 8))

epochs_range= range(10)

plotter_lib.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")

plotter_lib.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")

plotter_lib.axis(ymin=0.1,ymax=1)

plotter_lib.grid()

plotter_lib.title('Model Accuracy')

plotter_lib.ylabel('Accuracy')

plotter_lib.xlabel('Epochs')

plotter_lib.legend(['train', 'validation'])

# COMMAND ----------

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# COMMAND ----------

demo_resnet_model.save('/dbfs/team_j/model_checkpoint_ivan_three_class_Subset_Trial.h5')