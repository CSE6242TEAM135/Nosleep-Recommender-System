# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:52:39 2020

@author: karin
"""
import pandas as pd
import re
import string
import random
import nltk
import numpy as np
import os
import math
import string
import codecs
import json
from itertools import product
from inspect import getsourcefile
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

os.getcwd()
'C:\\Program Files\\PyScripter'
os.chdir(r"C:\\Users\\karin\\Documents\OMSA\\DVA 6242\\Group Project") 

###########################################
## [0.0] read in data and pre-processing ##
###########################################
comment_df = pd.read_csv('user_comments.csv')
comment_df['sentiment'] = comment_df['sentiment'].fillna(-99)
comment_df.head()
body = comment_df['body']
analyser = SentimentIntensityAnalyzer()

########################################
## [1.0] use Vader sentiment analysis ##
## 
## "Negative" = -1 
## "Neutral" = 0 
## "Positive" = 1 
## 
########################################
def sentiment_analyzer_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer() 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
    return sentiment_dict['compound']

test_df = comment_df[:200]
test_df =  test_df[["body", "score", "author_fullname", "id", "storyId", "author", 'sentiment' ]]
body = test_df["body"] 
test_df["compound"] = body.apply(sentiment_analyzer_scores)
compound = test_df["compound"]
test_df.head()
## comment_df["score"] = body.apply(sentiment_analyzer_scores)
## comment_df.head()
## assign vader score 
test_df['vader_sc'] = np.where(compound > 0, 1, (np.where(compound <0, -1, 0)))


## check Vader accuracy 
positive_com = test_df[test_df['sentiment'] == 1]
negative_com = test_df[test_df['sentiment'] == 0]
pos_count = np.sum(positive_com["sentiment"],axis=0)
pos_correct = np.sum(positive_com["vader_sc"] == 1, axis = 0)
neg_count = len(negative_com)
neg_correct = len(negative_com[negative_com["vader_sc"] == -1])

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

#################################
## [2.0] using StanfordCoreNLP ##
## Very negative" = 0 
## "Negative" = 1 
## "Neutral" = 2 
## "Positive" = 3 
## "Very positive" = 4

## set up StanfordCore NLP 
## cd stanford-corenlp-full-2018-10-05
## java -Xmx64m -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 -annotators tokenize,ssplit,pos,lemma,ner,parse,mention,coref
###################################

from pycorenlp import StanfordCoreNLP
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')

def get_sentiment(text):
    res = nlp_wrapper.annotate(text,
                       properties={'annotators': 'sentiment',
                                   'outputFormat': 'json',
                                   'timeout': 10000,
                       })
    print(text)
    print('Sentiment:', res['sentences'][0]['sentiment'])
    sentimentScore = res['sentences'][0]['sentimentValue']
    print('Sentiment score:', sentimentScore)
    return sentimentScore

test_df["stfd_sc"] = body.apply(get_sentiment)

## re-assign st score
test_df['NLP_sc'] = np.where(compound > 2, 1, (np.where(compound <2, -1, 0)))

st_pos_correct = np.sum(positive_com["NLP_sc"] == 1, axis = 0)
st_neg_correct = len(negative_com[negative_com["NLP_sc"] == -1])

print("Positive accuracy = {}% via {} samples".format(st_pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(st_neg_correct/neg_count*100.0, neg_count))


##########################################
## [3.0] compare vadar with StanfordNLP ##
##########################################
test_df["match"] = np.where(test_df["vader_sc"] == test_df["stfd_sc"], 1, 0)


##########################################
## [4.0] Collaborative filtering system ## 
## or nearest-neighbour-method
## or correlation analysis
##########################################


