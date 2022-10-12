# F-1
# import pip
# pip.main(['install', 'pyarrow'])  # for pd.read_parquet
# pip.main(['install', 'fastparquet'])  # for pd.read_parquet
# pip.main(['install', 'tensorflow'])
# pip.main(['install', 'tensorflow_addons'])
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.callbacks import EarlyStopping
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import tensorflow as tf
import itertools
import re
from sklearn.metrics import f1_score
import tensorflow_addons as tfa

nltk.download('stopwords')
nltk.download('wordnet')


pd.set_option('display.max_columns', None)  # display all columns
# pd.set_option('display.max_colwidth', None)  # expanding the dispay of every column
desired_width = 320
pd.set_option('display.width', desired_width)

data = pd.read_parquet(
    "C:/Users/Kaiyun Kang/Downloads/WebsiteTxt_0_True_True_fintent_mongo_cb_202203_c9e93402ee-search.parquet")
target = pd.read_parquet(
    "C:/Users/Kaiyun Kang/Downloads/SearchTopicFeatures_0_1_True_True_enterprise_software_fintent_mongo_cb_202203_13f6a2d73d-trainY.parquet")
target.reset_index(inplace=True)  # make domain_name as column, not index
print(data.head())
print(target.head(20))
print(data.columns)
print(target.columns)
print('data shape:', data.shape, '\ntarget shape:',
      target.shape)  # Don't have same number of entries, may need inner join to filter
# (23440, 8) (22987, 2)
data.drop('content_nav', axis=1, inplace=True)  # content_nav does not have any value

# EDA
print(data.info())  # check columns and Dtype
print(data.isna().sum())
print(data['status'].value_counts())  # all status is successful, so no need to worry
print(target.isna().sum())  # no NAN in target values
print(target['target'].value_counts())  # 0:1 = 2.5:1
print(target[target.target == 1])  # Look into target=1
print(data[data.domain_name == 'alchemistaccelerator.com'])

# different length of data and target
# inner join to get final df
df = data.merge(target, how='inner', on='domain_name')
df.set_index('domain_name',inplace=True)
print(df.shape)  # (22987, 9), same as target
print(df.head())
print(df.columns)
print(df['target'].value_counts())  # Imbalance, use F-1 seems appropriate as the evaluation metrics
print(df.isna().sum())

# Preprocess/Clean
cachedStopWords = stopwords.words('english')
toke = RegexpTokenizer(r'\w+')
lemma = WordNetLemmatizer()
def preprocess(text, cachedStopWords=cachedStopWords, toke=toke, lemma=lemma):
    # Remove numbers # Later tooknizer show that there are many numbers
    text0 = re.sub(r'\d+', '', text)
    # Tokenization while ignoring punctuations
    text1 = toke.tokenize(text0)
    # Lemmatisation and Lower casing
    text2 = [lemma.lemmatize(word.lower(), pos='v') for word in text1]
    # Removing Stop words
    text3 = [word for word in text2 if word not in cachedStopWords]
    return text3

data2 = df[['target']].copy() #df with preprocessed content_txt
#data3 = df['target'].copy() #df with preprocessed content_search
data4 = df[['target']].copy() #df with preprocessed website_summary(has 5506 nan)

data2['content_txt'] = df.content_txt.apply(lambda x: preprocess(x))
print(data2['content_txt'].apply(len).max())  # 127512
#data3['content_search'] = df.content_search.apply(lambda x: preprocess(x))
#print(data3['content_search'].apply(len).max()) # 127543
data4['website_summary'] = df.website_summary.copy()
data4.dropna(inplace=True)
print(data4.shape)
print(data4.isna().sum())
data4['website_summary'] = data4['website_summary'].apply(lambda x: preprocess(x))
print(data4['website_summary'].apply(len).max()) #365

path1 = 'E:/nlp/data2.parquet'
#path2 = 'E:/nlp/data3.parquet'
path3 = 'E:/nlp/data4.parquet'
data2.to_parquet(path1)
#data3.to_parquet(path2)
data4.to_parquet(path3)

