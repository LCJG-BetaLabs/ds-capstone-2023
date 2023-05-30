# Databricks notebook source
# Need to Keep this code --- Update installed library 
!pip install --upgrade numpy
!pip install --upgrade gensim
# %pip install --ignore-installed spacy
%pip install -U spacy

# Small package, inferior quality of word vector
!python -m spacy download en_core_web_sm 

# large package, superior quality of word vector
!python -m spacy download en_core_web_lg

# Accuracy package, best quality of word vector, required CuDF
#!python -m spacy download en_core_web_trf
# import en_core_web_trf

# COMMAND ----------

# Import library
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# Storage path for LC data
container = "data3" 
data_path = f"abfss://{container}@capstone2023cuhk.dfs.core.windows.net/"

# Storage path for teams
team_container = "capstone2023-cuhk-team-f"
team_path = f"abfss://{team_container}@capstone2023cuhk.dfs.core.windows.net/"

# COMMAND ----------

# MAGIC %md
# MAGIC <b>From Team Space to Local</b>

# COMMAND ----------

# copy folder from ABFS to local
dbutils.fs.cp(os.path.join(team_path), "file:/local", recurse=True) 

# To list out file/dirs
display(dbutils.fs.ls("file:/local"))
print("=========================================================================================")
display(dbutils.fs.ls(os.path.join("file:/local","competitor_analysis")))

# COMMAND ----------

# Data - LC
LC = pd.read_csv(os.path.join("file:/local","LC_attribute.csv")).reset_index()
print(LC.shape);print(f"Unique ID: {LC['atg_code'].nunique()==len(LC)}")
display(LC)

# COMMAND ----------

# Data - FF
FF = pd.read_csv(os.path.join("file:/local","farfetch_KN230402.csv")).drop(columns=['Unnamed: 0']).reset_index()
print(FF.shape);print(f"Unique ID: {FF['ProductID'].nunique()==len(FF)}")
display(FF)

# COMMAND ----------

# Data - NAP
NAP=pd.read_csv(os.path.join("file:/local","netaporter_merged_sourcedf.csv")).drop(columns=['index','Unnamed: 0']).reset_index()
NAP['Colors']=NAP['Colors'].fillna("NA").apply(lambda x: x.translate(str.maketrans("","","[']")).split(','))
print(NAP.shape);print(f"Unique ID: {NAP['ProductID'].nunique()==len(NAP)}")
display(NAP)

# COMMAND ----------

display(LC.head(1))
display(FF.head(1))
display(NAP.head(1))

# COMMAND ----------

# MAGIC %md
# MAGIC Potential Variables to Match
# MAGIC - Desc / Product Name
# MAGIC - Brand Name
# MAGIC - Long Desc
# MAGIC - Color
# MAGIC - Composition
# MAGIC - Keywords from Long Desc, Color, Composition
# MAGIC - Product Image
# MAGIC
# MAGIC Less Potential Variables to Match
# MAGIC - Care(?)
# MAGIC - Size and Fit

# COMMAND ----------

# MAGIC %md
# MAGIC <b>Text Vectorization</b>
# MAGIC - Bag of Word / CountVector (sklearn)
# MAGIC - TF-IDF Vector (sklearn)
# MAGIC - Word2Vec (Spacy)
# MAGIC - Doc2Vec (Genism)

# COMMAND ----------

# bag of words = assessing the lexical similarity of text, i.e., how similar documents are on a word level.
from sklearn.feature_extraction.text import CountVectorizer

# tf-idf = bag of word model taken into count appearance of terms in all documents
from sklearn.feature_extraction.text import TfidfVectorizer

# Word2Vec = vector for semantic meaning of tokens
# Doc vector is just an average meaning representation of the token vectors.
# Documentation: https://spacy.io/usage
import spacy
import en_core_web_sm
import en_core_web_lg

# Doc2Vec = vector for semantic meaning of paragraphs
# Documentation: https://radimrehurek.com/gensim/index.html
# Doc2Vec documentation: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
import gensim

# COMMAND ----------

# spacy - Introduction
nlp=en_core_web_lg.load() #nlp=spacy.load("en_core_web_lg")
print(nlp.pipe_names)

text_LC=LC['long_desc']
doc=nlp(text_LC[1]) # doc=nlp("Yesterday was a good day")

# illustrate attributes of tokens and named entities
display(pd.DataFrame(columns=['text','lemma_','is_stop','pos_','tag_','dep_','is_alpha'],
                     data=[(token.text,token.lemma_,token.is_stop,token.pos_,token.tag_, token.dep_,token.is_alpha) for token in doc]))
display(pd.DataFrame(columns=['text','ent_label'],
                     data=[(ent.text,ent.label_) for ent in doc.ents][0:5]))

# List of pos: https://universaldependencies.org/u/pos/
# Detailed list of pos: https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/

# COMMAND ----------

nlp=en_core_web_lg.load() #nlp=spacy.load("en_core_web_lg")
print(nlp.pipe_names)

def clean_corpus(corpus,nlp=nlp):
    docs=list(nlp.pipe(corpus))
    # "VB","VBP","VBZ" means verbs in present tense
    corpus_cleaned=[' '.join([token.lemma_ for token in doc 
                           if not (token.is_stop or token.pos_ in ["PUNCT","SYM","X","INTJ","ADV"] or token.tag_ in ["VB","VBP","VBZ"])])
                     for doc in docs]
    
    return pd.Series(corpus_cleaned)
  
def best_match(vectors_A,vectors_B,Group_A=["LC",LC,'atg_code'],Group_B=['NAP',NAP,'ProductID']):
    # define variables
    name_A=Group_A[0];df_A=Group_A[1];idx_A=Group_A[2]
    name_B=Group_B[0];df_B=Group_B[1];idx_B=Group_B[2]
    
    # cosine similarity matrix
    similarity=pd.DataFrame(cosine_similarity(vectors_A,vectors_B))
    
    # find best match
    similarity_match=similarity.idxmax(axis=1)
    similarity_match=pd.DataFrame({'A loc':similarity_match.index,'B loc':similarity_match,'similarity':similarity.max(axis=1)})
    
    # label with original id
    df_A['index']=range(len(df_A));df_B['index']=range(len(df_B))
    similarity_match=similarity_match.merge(df_A[['index',idx_A]],how='left',left_on='A loc',right_on='index')
    similarity_match=similarity_match.merge(df_B[['index',idx_B]],how='left',left_on='B loc',right_on='index')
    similarity_match=similarity_match[['A loc','B loc',idx_A,idx_B,'similarity']]
    similarity_match.columns=[f'{name_A} loc',f'{name_B} loc',f'{name_A} id (best)',f'{name_B} id (best)','similarity (best)']

    return similarity_match

def similarity_scores(vectors_A,vectors_B,Group_A=["LC",LC,'atg_code'],Group_B=['NAP',NAP,'ProductID'],vec_source="(source)",vec_type="(vectorizer)"):
    # define variables
    name_A=Group_A[0];df_A=Group_A[1];idx_A=Group_A[2]
    name_B=Group_B[0];df_B=Group_B[1];idx_B=Group_B[2]
    
    # cosine similarity matrix
    similarity=pd.DataFrame(cosine_similarity(vectors_A,vectors_B))
    
    # change to long form table
    similarity['index']=range(len(similarity))
    score_table=similarity.melt(id_vars='index',var_name='competitor loc',value_name='similarity')
    score_table.columns=['A loc','B loc','similarity']
        
    # label with original id
    df_A['index']=range(len(df_A));df_B['index']=range(len(df_B))
    score_table=score_table.merge(df_A[['index',idx_A]],how='left',left_on='A loc',right_on='index')
    score_table=score_table.merge(df_B[['index',idx_B]],how='left',left_on='B loc',right_on='index')
    score_table=score_table[['A loc','B loc',idx_A,idx_B,'similarity']]
    score_table.columns=[f'{name_A} loc',f'{name_B} loc',f'{name_A} id',f'{name_B} id',f'similarity_{vec_source}_{vec_type}']
    
    return score_table

# COMMAND ----------

# Clean Text for Analysis
text_LC=clean_corpus(LC['long_desc'])
text_NAP=clean_corpus(NAP['Editor Notes'].fillna(""))

text_all=pd.concat([text_LC,text_NAP])
print(f"Count: text_LC={len(text_LC)}; text_NAP={len(text_NAP)}; text_all={len(text_all)}")

# COMMAND ----------

# Text vectorization - CountVectorizer
wob_vectorizer=CountVectorizer()

vectors_all=wob_vectorizer.fit_transform(text_all)
vectors_all=vectors_all.toarray()
print(vectors_all.shape)

vectors_LC=vectors_all[:len(text_LC)]
print(vectors_LC.shape)
vectors_NAP=vectors_all[len(text_LC):(len(text_LC)+len(text_NAP))]
print(vectors_NAP.shape)

# Show best match only
display(best_match(vectors_LC,vectors_NAP,["LC",LC,'atg_code'],['NAP',NAP,'ProductID']))

# Show all similarity scores
score_table=similarity_scores(vectors_LC,vectors_NAP,["LC",LC,'atg_code'],['NAP',NAP,'ProductID'],vec_source="LongDesc",vec_type="WOB")
display(score_table)

# Only show rows with highest similarity; Essentially same effect as function "best_match"
display(score_table.loc[score_table.groupby('LC loc')['similarity_LongDesc_WOB'].idxmax()])

# COMMAND ----------

# Text vectorization - TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer()

vectors_all=tfidf_vectorizer.fit_transform(text_all)
vectors_all=vectors_all.toarray()
print(vectors_all.shape)

vectors_LC=vectors_all[:len(text_LC)]
print(vectors_LC.shape)
vectors_NAP=vectors_all[len(text_LC):(len(text_LC)+len(text_NAP))]
print(vectors_NAP.shape)

best_match(vectors_LC,vectors_NAP,["LC",LC,'atg_code'],['NAP',NAP,'ProductID'])

# COMMAND ----------


# Text vectorization - Word2Vec (spacy)
def Word2Vec(corpus,nlp=nlp):
    docs=list(nlp.pipe(corpus))
    vectors=np.array([doc.vector for doc in docs])
    return vectors

# COMMAND ----------

vectors_LC=Word2Vec(clean_corpus(text_LC))
print(vectors_LC.shape)

vectors_NAP=Word2Vec(clean_corpus(text_NAP))
print(vectors_NAP.shape)

best_match(vectors_LC,vectors_NAP,["LC",LC,'atg_code'],['NAP',NAP,'ProductID'])

# COMMAND ----------

# Text Vectorization - Doc2Vec (genism)
def read_corpus(corpus, tokens_only=False):
    for i, doc in enumerate(corpus):
        #tokens = gensim.utils.simple_preprocess(doc)
        doc=nlp(doc)
        tokens=[token.lemma_ for token in doc
                if not (token.is_stop or token.pos_ in ["PUNCT","SYM","X","INTJ","ADV"] or token.tag_ in ["VB","VBP","VBZ"])]
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def Doc2Vec(corpus):
    train_corpus = list(read_corpus(corpus))
    test_corpus = list(read_corpus(corpus, tokens_only=True))

    # Only words with at least 2 occurences are considered
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    vectors = np.array([model.infer_vector(tokens) for tokens in test_corpus])
    return vectors

# COMMAND ----------

vectors_LC=Doc2Vec(text_LC)
print(vectors_LC.shape)

vectors_NAP=Doc2Vec(text_NAP)
print(vectors_NAP.shape)

best_match(vectors_LC,vectors_NAP,["LC",LC,'atg_code'],['NAP',NAP,'ProductID'])

# COMMAND ----------

# MAGIC %md
# MAGIC ==========================================<br>
# MAGIC <b>Code for Reference Only</b>

# COMMAND ----------

# Define data
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]

# Create a PySpark DataFrame
df = spark.createDataFrame(data, ["Name", "Age"])

# Write parquet from storage
df.write.mode('overwirte').parquet(os.path.join(team_path, "test_parquet"))

# Read parquet from storage
read_df = spark.read.parquet(path)
display(read_df)

# COMMAND ----------

# Saving model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
x_train = np.random.rand(100, 3)
y_train = np.random.randint(2, size=100)
all_features = tf.keras.Input(shape = x_train.shape[1])
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(all_features, output)
model.compile(loss="binary_crossentropy", 
             optimizer=Adam(),
             metrics=["accuracy"])

model.fit(x=x_train, y=y_train, epochs=10)
data = model.save("file:/tmp/test_model")
dbutils.fs.cp("file:/tmp/test_model", os.path.join(team_path, "test_model"), recurse=True) # copy folder from local to ABFS
d
# load model
dbutils.fs.cp(os.path.join(team_path, "test_model"), "file:/tmp/test_model", recurse=True) # copy folder from ABFS to local
model = tf.keras.models.load_model("file:/tmp/test_model")

# COMMAND ----------

# MAGIC %md
# MAGIC <b>From Local to Team Space (Completed)</b>

# COMMAND ----------

# copy folder from local to ABFS
# dbutils.fs.cp("file:/competitor_analysis", os.path.join(team_path, "competitor_analysis"), recurse=True)

# To list out file/dirs in team space
dbutils.fs.ls(os.path.join(team_path, "competitor_analysis"))

# COMMAND ----------

# MAGIC %md
# MAGIC <b>From LC Data Space to Local</b>

# COMMAND ----------

# Spark - Direct Read from LC data space
df = spark.read.format("csv").load(os.path.join(data_path, "competitor_analysis", "attribute.csv"))
df.show()