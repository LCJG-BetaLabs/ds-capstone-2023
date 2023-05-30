# Databricks notebook source
import pandas as pd
import os
import numpy as np

# COMMAND ----------

#resampling method 3
test_df = pd.read_csv(os.path.join("file:/dbfs", "final_test_data3.csv"))
train_df = pd.read_csv(os.path.join("file:/dbfs", "final_train_data3.csv"))
val_df = pd.read_csv(os.path.join("file:/dbfs", "final_val_data3.csv"))

# COMMAND ----------

test_df['class'].unique()

# COMMAND ----------

# MAGIC %sh
# MAGIC /databricks/python3/bin/pip install spacy 
# MAGIC /databricks/python3/bin/python3 -m spacy download en_core_web_sm

# COMMAND ----------

# DBTITLE 1,Text Preprocessing
# Clean the text by removing stopwords and punctuation
import nltk
nltk.download('stopwords') # Download the stopwords data
nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from nltk.stem import 	WordNetLemmatizer
#from nltk import pos_tag

import spacy
nlp = spacy.load('en_core_web_sm')

import string

stop_words = set(stopwords.words('english'))

def clean_stopwords(text):
    # Define a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    # Use the translation table to remove punctuation from the text
    text = text.translate(translator)

    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if not word in stop_words]

    filtered_text = ' '.join(filtered_tokens) 
    return filtered_text  
"""
def lemmatize(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = []
    for word,tag in pos_tag(tokens):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            lemma = word
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, wntag)
        lemmatized_tokens.append(lemma)
    return ' '.join(lemmatized_tokens)
"""
def lemmatize(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        tokens.append(token)
    return " ".join([token.lemma_ for token in doc])



# COMMAND ----------

clean_stopwords("This is an example sentence that contains some stop words.")

# COMMAND ----------

lemmatize("crop top is cropped")

# COMMAND ----------

train_df['purpose'] = 'train'
val_df['purpose'] = 'val'
test_df['purpose'] = 'test'

# COMMAND ----------

test_df['class'].unique()

# COMMAND ----------

combined_df = pd.concat([train_df,val_df,test_df])

# COMMAND ----------

combined_df['combined_desc'] = combined_df['color_desc'] + " " + combined_df['prod_desc_eng'] + " " + combined_df['long_desc']

# COMMAND ----------

combined_df['cleaned_combined_desc'] = combined_df['combined_desc'].apply(lemmatize)

# COMMAND ----------

combined_df.shape

# COMMAND ----------

X = np.array(combined_df['cleaned_combined_desc'].tolist())
y = np.array(combined_df['class'].tolist())

# COMMAND ----------

# DBTITLE 1,Check if IDF should be used
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([('tfidf',TfidfVectorizer()),
                     ('sgd',SGDClassifier())])
params = {'tfidf__use_idf':(False,True)}
gridsearch = GridSearchCV(pipeline,params)
gridsearch.fit(X,y)
print(gridsearch.best_params_)

# COMMAND ----------

import tensorflow.keras as K 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Convert extracted keywords into feature vectors
vectorizer = TfidfVectorizer(stop_words='english'
,use_idf=False
,min_df=0.05
,max_df=0.6
,max_features=128
,ngram_range=(1,2)
,sublinear_tf=True)
X_transformed = vectorizer.fit_transform(X).toarray()
print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names())


y_transformed = K.utils.to_categorical(y)

# COMMAND ----------

def get_extracted_kwds(text):
    kwd_list = []
    for word in text.split():
        if word in ' '.join(vectorizer.get_feature_names()):
            kwd_list.append(word)
    return ' '.join(kwd_list)


# COMMAND ----------

tfidf_df_without_7 = combined_df[combined_df['class']!=7]

# COMMAND ----------

tfidf_df_without_7['extracted_kwds'] = tfidf_df_without_7['cleaned_combined_desc'].apply(get_extracted_kwds)

# COMMAND ----------

tfidf_df_without_7['atg_code'].iloc[9]

# COMMAND ----------

tfidf_df_without_7['cleaned_combined_desc'].iloc[40]

# COMMAND ----------

lemmatize(tfidf_df_without_7['cleaned_combined_desc'].iloc[40])

# COMMAND ----------

tfidf_df_without_7['extracted_kwds'].iloc[10]

# COMMAND ----------

len(X_transformed),len(y_transformed)

# COMMAND ----------

tfidf_df_without_7['X_transformed'] = X_transformed.tolist()
tfidf_df_without_7['y_transformed'] = y_transformed.tolist()

# COMMAND ----------

y_train = np.array(tfidf_df_without_7[(tfidf_df_without_7['purpose']=='train')]['y_transformed'].tolist())
y_val = np.array(tfidf_df_without_7[(tfidf_df_without_7['purpose']=='val')]['y_transformed'].tolist())
y_test = np.array(tfidf_df_without_7[(tfidf_df_without_7['purpose']=='test')]['y_transformed'].tolist())

X_train = np.array(tfidf_df_without_7[(tfidf_df_without_7['purpose']=='train')]['X_transformed'].tolist())
X_val = np.array(tfidf_df_without_7[(tfidf_df_without_7['purpose']=='val')]['X_transformed'].tolist())
X_test = np.array(tfidf_df_without_7[(tfidf_df_without_7['purpose']=='test')]['X_transformed'].tolist())


# COMMAND ----------

y_train.shape,y_val.shape,y_test.shape

# COMMAND ----------

X_train.shape,X_val.shape,X_test.shape

# COMMAND ----------

# DBTITLE 1,Use sequential NN to train: second attempt
from sklearn.metrics import accuracy_score

FILEPATH = "/dbfs/text_model_v1"

# Early stopping  
check_point = K.callbacks.ModelCheckpoint(filepath=FILEPATH,
                                              monitor="val_accuracy",
                                              mode="max",
                                              save_best_only=True,
                                              )

# Define neural network architecture
input_shape = (X_train.shape[1],)
num_classes = y_train.shape[1]
model = K.models.Sequential()
model.add(K.layers.Dense(32, input_shape = input_shape, activation="relu"))
#model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, batch_size=8, epochs=10, verbose=1, validation_data=(X_val, y_val),callbacks=[check_point])

model.summary()
model.save(FILEPATH)


# COMMAND ----------

# Evaluate the model on test data
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
print(list(zip(test_df['class'].tolist(),y_pred_class)))

# COMMAND ----------

from sklearn import metrics

# confusion matrix
confusion_matrix = metrics.confusion_matrix(test_df['class'].tolist(), y_pred_class)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

# COMMAND ----------

#get probabilities on test data
test_prob_data = pd.DataFrame(y_pred,columns=['Class1','Class2','Class3','Class4','Class5','Class6'])
test_prob_data['pred'] = y_pred_class.tolist()
test_prob_data['atg_code'] = test_df['atg_code'].tolist()
test_prob_data['true_label'] = test_df['class'].tolist()
test_prob_data['extracted_kwds'] = tfidf_df_without_7[tfidf_df_without_7['purpose']=='test']['extracted_kwds']
test_prob_data=test_prob_data[['atg_code','extracted_kwds','true_label','pred','Class1','Class2','Class3','Class4','Class5','Class6']]
test_prob_data.head(20)


# COMMAND ----------

test_prob_data=test_prob_data[['atg_code','pred','Class1','Class2','Class3','Class4','Class5','Class6']]
test_prob_data.to_csv('/dbfs/tfidf_test_prob_data.csv', index=False)

# COMMAND ----------

# DBTITLE 1,Run model on untagged images
df_cleansed = pd.read_csv(os.path.join("file:/dbfs", "cleansed_data.csv"))
df_class7 = df_cleansed.loc[df_cleansed['class']==7]
df_class7.head()

# COMMAND ----------

# DBTITLE 1,Preprocess untagged data
df_class7['combined_desc'] = df_class7['color_desc'] + " " + df_class7['prod_desc_eng'] + " " + df_class7['long_desc']
df_class7['cleaned_combined_desc'] = df_class7['combined_desc'].apply(lemmatize)

# COMMAND ----------

df_class7.head()

# COMMAND ----------

X_class7 = np.array(df_class7['cleaned_combined_desc'].tolist())

# COMMAND ----------

# Convert extracted keywords into feature vectors
vectorizer = TfidfVectorizer(stop_words='english'
,use_idf=False
,min_df=0.05
,max_df=0.6
,ngram_range=(1,2)
,sublinear_tf=True
,max_features=128
)
X_class7_transformed = vectorizer.fit_transform(X_class7).toarray()
print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names())

# COMMAND ----------

df_class7.shape

# COMMAND ----------

len(X_class7_transformed)

# COMMAND ----------

#X_class7 = df.merge(df_embs_original, on=['atg_code'], how='inner').iloc[:,1:-1].to_numpy()
y_pred_class7 = model.predict(X_class7_transformed)
y_pred_class_class7 = np.argmax(y_pred_class7, axis=1)

# COMMAND ----------

len(y_pred_class7)

# COMMAND ----------

df['pred'] = y_pred_class_class7
df.groupby('pred').size()

# COMMAND ----------

len(y_pred_class_class7)

# COMMAND ----------

df_class7['extracted_kwds'] = df_class7['cleaned_combined_desc'].apply(get_extracted_kwds)

# COMMAND ----------

#get probabilities on untagged data
untagged_prob_data = pd.DataFrame(y_pred_class7,columns=['Class1','Class2','Class3','Class4','Class5','Class6'])
untagged_prob_data['pred'] = y_pred_class_class7.tolist()
untagged_prob_data['atg_code'] = df_class7['atg_code'].tolist()
untagged_prob_data['extracted_kwds'] = df_class7['extracted_kwds'].tolist()
untagged_prob_data=untagged_prob_data[['atg_code','extracted_kwds','pred','Class1','Class2','Class3','Class4','Class5','Class6']]
#untagged_prob_data.isnull().values.any()
untagged_prob_data.head(10)


# COMMAND ----------

untagged_prob_data[['atg_code','pred','Class1','Class2','Class3','Class4','Class5','Class6']].to_csv('/dbfs/tfidf_untagged_prob_data.csv', index=False)

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def img_path(atg_code):
    return f"/dbfs/image/{atg_code}_in_xl.jpg"

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

for idx, items in untagged_prob_data.loc[untagged_prob_data['pred']==0].sample(n=10).iterrows():
    display_image(img_path(items.atg_code))

# COMMAND ----------

 