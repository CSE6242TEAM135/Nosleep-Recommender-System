# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:46:08 2020

@author: karin
"""
      
## Tokenization: Split the text into sentences and the sentences into words. 
## Lowercase the words and remove punctuation.
## Words that have fewer than 3 characters are removed.
## All stopwords are removed.
## Words are lemmatized — words in third person are changed to first person 
## and verbs in past and future tenses are changed into present.
## Words are stemmed 
        
        
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

stemmer = PorterStemmer() 

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = comment_df["comment_body"].map(preprocess)

## Bag of Words on the Data set
## Create a dictionary from ‘processed_docs’ 
## count the number of times a word appears in data
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

## Filter out tokens that appear in
## less than 15 documents (absolute number) or
## more than 0.5 documents (fraction of total corpus size, not absolute number).
## after the above two steps, keep only the first 100000 most frequent tokens.
        
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)