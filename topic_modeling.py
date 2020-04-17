import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import string
from operator import itemgetter
import re
from scipy.spatial import distance_matrix

import nltk
nltk.download('stopwords')
nltk.download('names')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.corpus import names
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 

import gensim
import gensim.corpora as corpora
from gensim.models import HdpModel
from gensim.utils import lemmatize





# Read in the data
d = {"author": [], "author_fullname": [], "full_link": [], "id": [], "score": [], "selftext": [], "title": [], "sortKey": []}
file_location = "your_file_here"
with open(file_location) as f:
        for line in f:
            line = ast.literal_eval(line)
            d["author"].append(line["author"])
            d["author_fullname"].append(line["author_fullname"])
            d["full_link"].append(line["full_link"])
            d["id"].append(line["id"])
            d["score"].append(line["score"])
            d["selftext"].append(line["selftext"])
            d["title"].append(line["title"])
            d["sortKey"].append(line["sortKey"])

story_df = pd.DataFrame(data=d)

# Preprocessing helper function
def preprocess(df,col):
    print("Removing punctuation")
    df['processed'] = df[col].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    
    print("Making all lower case")
    df['processed'] = df['processed'].map(lambda x: x.lower())
    
    print("Removing stop words and names")
    combined_stops = names.words('male.txt')
    combined_stops.extend(stopwords.words('english'))
    combined_stops.extend(names.words('female.txt'))
    combined_stops = set(combined_stops)
    df['processed'] = df['processed'].map(lambda x: ' '.join(y for y in x.split() if y not in combined_stops))
    
    print("Stemming")
    stemmer = PorterStemmer() 
    df['processed'] = df['processed'].map(lambda x: [stemmer.stem(y) for y in x.split()])
    
    print("Removing numbers")
    df['processed'] = df['processed'].map(lambda x: [item for item in x if not item.isdigit()])
    
    print("putting processed text back together")
    df['processed_text'] = df['processed'].map(lambda x: " ".join(x))
    
    return df

story_df = preprocess(story_df, 'selftext')

# Train the hdp model

# Create Dictionary
id2word = corpora.Dictionary(story_df['processed'])

# Create Corpus
texts = story_df['processed']

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

hdpmodel = HdpModel(corpus=corpus, id2word=id2word)

# Helper function for making a dataframe of the topic probabilities
def make_topic_df(hdpmodel,corpus):
    topic_df = pd.DataFrame()
    for i in range(len(corpus)):
        if i % 5000 == 0:
            print(i)
        doc_hdp = hdpmodel[corpus[i]]
        if doc_hdp:
            for x,y in doc_hdp:
                topic_df.at[i,x] = y
        else:
            topic_df.at[i,9999] = 1.0

    topic_df = topic_df.fillna(0)
    return topic_df

topic_df = make_topic_df(hdpmodel,corpus)

# Make a distance matrix from the topic_df
dist_mat = distance_matrix(topic_df,topic_df)

# Make a dictionary where each story id has it's 200 closest stories by topic
distance_dict = defaultdict(list)
for i in range(len(dist_mat)):
    if i % 5000 == 0:
        print(i)
    ind = np.argpartition(dist_mat[i], 201)[:201]
    ids = [story_df['id'][x] for x in ind if x != i]
    distance_dict[story_df['id'][i]] = ids

# Export to a JSON
with open('distance_dict.txt', 'w') as file:
     file.write(json.dumps(distance_dict))