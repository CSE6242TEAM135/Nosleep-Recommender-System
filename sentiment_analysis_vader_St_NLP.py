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

df = (comment_df['body'].str.len() < 100)
df100 = comment_df.loc[df]
## df100 = df100[:200] 
## body = comment_df['body']
## analyser = SentimentIntensityAnalyzer()

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
test_df = df100
test_df = comment_df[:2000]
test_df =  test_df[["body", "score", "author_fullname", "id", "storyId", "author", 'sentiment', 'prediction' ]]
body = test_df["body"] 
test_df["compound"] = body.apply(sentiment_analyzer_scores)
compound = test_df["compound"]
test_df.head()
## comment_df["score"] = body.apply(sentiment_analyzer_scores)
## comment_df.head()
## assign vader score 
test_df['vader_sc'] = np.where(compound > 0, 1, (np.where(compound <=0, -1, 0)))


## check Vader accuracy 
positive_com = test_df[test_df['sentiment'] == 1]
negative_com = test_df[test_df['sentiment'] == 0]
pos_count = np.sum(positive_com["sentiment"],axis=0)
pos_correct = np.sum(positive_com["vader_sc"] == 1, axis = 0)
neg_count = len(negative_com)
neg_correct = len(negative_com[negative_com["vader_sc"] == -1])

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))


sentiment_analyzer_scores("This just keeps getting better")

#################################
## [2.0] using StanfordCoreNLP ##
## Very negative" = 0 
## "Negative" = 1 
## "Neutral" = 2 
## "Positive" = 3 
## "Very positive" = 4

## set up StanfordCore NLP 
## cd stanford-corenlp-full-2018-10-05
## java -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 20000 -annotators tokenize,ssplit,pos,lemma,ner,parse,coref
###################################

from pycorenlp import StanfordCoreNLP
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')

def get_sentiment(text):
    res = nlp_wrapper.annotate(text,
                       properties={'annotators': 'sentiment',
                                   'outputFormat': 'json',
                                   'timeout': 18000,
                       })
    ## print(text)
    ## print('Sentiment:', res['sentences'][0]['sentiment'])
    sentimentScore = res['sentences'][0]['sentimentValue']
    ## print('Sentiment score:', sentimentScore)
    return sentimentScore

test_df["stfd_sc"] = body.apply(get_sentiment)
stfd_pred = test_df["stfd_sc"]
stfd_pred = pd.to_numeric(stfd_pred)

## re-assign st score
## test_df['NLP_sc'] = np.where(stfd_pred > 2, 1, (np.where(stfd_pred <2, -1, 0)))
test_df['NLP_sc'] = np.where(stfd_pred > 2, 1, -1)

positive_com = test_df[test_df['sentiment'] == 1]
negative_com = test_df[test_df['sentiment'] == 0]

st_pos_correct = np.sum(positive_com["NLP_sc"] == 1, axis = 0)
st_neg_correct = len(negative_com[negative_com["NLP_sc"] == -1])

print("Positive accuracy = {}% via {} samples".format(st_pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(st_neg_correct/neg_count*100.0, neg_count))


##########################################
## [3.0] compare vadar with StanfordNLP ##
##########################################
test_df["match"] = np.where(test_df["vader_sc"] == test_df["stfd_sc"], 1, 0)

positive_com.to_clipboard(sep='\t')

test_df.to_clipboard(sep='\t')

##############################
## [4.0] Recommender system ##
##
##############################
import matplotlib.pyplot as plt
import seaborn as sns

positive_com["rating"] = pd.to_numeric(stfd_pred)
positive_com.groupby('storyId')['rating'].mean().head()

ratings_mean_cnt = pd.DataFrame(positive_com.groupby('storyId')['rating'].mean())
ratings_mean_cnt['rating_cnt'] = pd.DataFrame(positive_com.groupby('storyId')['rating'].count())
ratings_mean_cnt.head()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
%matplotlib inline

#########################################################
## [4.1] Histogram
## plot a histogram for the no. of ratings
## the plot shows most of the stories receive 1 rating ##
## num of movies recieve 2 or 3 ratings are low. 
#########################################################
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_cnt['rating_cnt'].hist(bins=50)

## plot histgram on avg rating 
## data has a weak normal distribution with the mean of around 3.5. 
## There are a few outliers in the data.

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_cnt['rating'].hist(bins=50)

## storeis with a higher no of ratings usually have a high average rating as well
## since a good story is usually popular and read by a large no of people 
## therefore usually has a higher rating

## movies with higher average ratings actually have more number of ratings, 
## compared with movies that have lower average ratings.

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='rating', y = 'rating_cnt', data = ratings_mean_cnt, alpha=0.4)

################################################
## [5.0] Finding Similarities Between Stories ##
################################################

## use the correlation between the ratings of a movie as the similarity metric. 
## this matrix has a lot of null values 
## since every story is not rated/commented by every user.
##
story_rating = positive_com.pivot_table(index='id', columns='storyId', values='rating')
story_rating.head()

example_ratings = story_rating['a0vz3f']

similar_story = story_rating.corrwith(example_ratings)
corr_example = pd.DataFrame(similar_story, columns=['Correlation'])
similar_story.dropna(inplace = True)
similar_story.head()