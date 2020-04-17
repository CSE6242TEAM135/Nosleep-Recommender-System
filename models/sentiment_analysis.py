#The following tutorial at https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
#has the author Shaumik Daityari showing how to do sentiment analysis using the nltk toolkit for python. This was used to create a Naive 
#Classifier model using the comment datasets for NoSleep. The final model was built on using training data for about 1250 comments.

#https://www.nltk.org/api/nltk.tokenize.html was also used
import pandas as pd
import re
import string
import random
import nltk
import json
import numpy as np
import math

#Note that the following packages need to be installed using pip or similar processes
#1) vader
#2) nltk
#Please note that when running this for the 1st time, the loadModules function must also be run
#as this downloads NLTK modules that are needed to set everything up.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier

class SentimentAnalysis_Modeler:
    #This initializes variables based on the small sample size provided. Note in practice
    #you will want to manually rate thousands of comments, among other things
    #like setting up bigrams, in order to create a robust model
    def __init__(self):
        #training_data - No. of rows for the Training Data set whose comments were manually rated
        #test_data - No. of rows for the Testing Data set whose comments were manually rated        
        #train_ratio - % of data the Model will used for training
        self.training_data_records = 121
        self.test_data_records = 100
        self.train_ratio = 0.7
    
    #This function only has to be run once. It downloads the modules that are necessary to perform sentiment analysis
    #using NLTK.
    def loadModules():
        #Details can be viewed at the following:
        #https://www.nltk.org/_modules/nltk/tokenize/punkt.html
        #https://www.nltk.org/howto/wordnet.html
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')    
        
    def loadComments(self):
        #This loads the entire user Comments into a Panda DataFrame
        #Note it has to be in the same directory as the notebook unless change
        #This sets all predictions to -1. Note here that you could choose to load the non-training
        #non-testing data seperately
        self.comment_full = pd.read_csv('sample_files\\user_comments_sample.csv')
        self.comment_full['prediction'] = -1
        
        #Sentiment stores the manual prediction which is used later on for testing
        #the accuracy and recall of the model. Note here that you could create and use another
        #field, but will have to update everything that references "sentiment"
        self.comment_full['sentiment'] = self.comment_full['sentiment'].fillna(-1)       
        
        #This creates a training and testing data set using the variables assigned in __init__
        #The No. of test data is converted into an actual position
        testing_data_pos = self.training_data_records + self.test_data_records
        self.comment_df = self.comment_full[:self.training_data_records]
        self.random_df = self.comment_full[self.training_data_records:testing_data_pos]
        
        #This resets the index used by the random_df
        self.random_df = self.random_df.reset_index(drop=True)
     
    #This section cleans up the code by removing false comments as well as comments that provide
    #little value (eg/ < 5 characters). The exception is key words like more/moar etc which indicates
    #the story was good.
    #THIS ONLY NEEDS TO BE RUN ONCE AS IT SAVES THE INFO BACK IN THE ORIGINAL FILE
    def preCleaning(self):
        comment_full = pd.read_csv('sample_files\\user_comments_sample.csv')
        
        comment_df = comment_full
        comment_df['body'] = comment_df['body'].str.strip()
        
        #Removes all comments that have the term 'remove' in it. Typically this is because the Comment was
        #removed by the moderators. Note this could be refined further
        comment_c = comment_df[~comment_df['body'].str.contains('remove')]
        
        #Removes all comments that talk about 'cake day' as this usually celebrates a yearly milestone from
        #when the author or someone else joins reddit
        comment_c = comment_c[~(comment_c['body'].str.lower()).str.contains('cake day')]
        
        #Remove all comments with the phrases below as they aren't talking about the story at all
        comment_c = comment_c[~comment_c['body'].str.contains('It looks like there may be more to this story')]
        comment_c = comment_c[~comment_c['body'].str.contains('https://red')]
        
        #Removes all comments that have a lenght less than 5 characters
        comment_c = comment_c[comment_c['body'].str.len() >= 5]
        
        #Keep all instances of 'more', 'moar', 'sick', 'holy' as they typically represent "good" comments
        comment_more = comment_df[(comment_df['body'].str.len() < 5) & ((comment_df['body'].str.lower()).str.contains('more'))]
        comment_more["sentiment"] = 3
        comment_moar = comment_df[(comment_df['body'].str.len() < 5) & ((comment_df['body'].str.lower()).str.contains('moar'))]
        comment_moar["sentiment"] = 3
        comment_sick = comment_df[(comment_df['body'].str.len() < 5) & ((comment_df['body'].str.lower()).str.contains('sick'))]
        comment_sick["sentiment"] = 3
        comment_holy = comment_df[(comment_df['body'].str.len() < 5) & ((comment_df['body'].str.lower()).str.contains('holy'))]
        comment_holy["sentiment"] = 3
        
        #This concatanates all of the comment datasets together
        comment_c = pd.concat([comment_c, comment_more])
        comment_c = pd.concat([comment_c, comment_moar])
        comment_c = pd.concat([comment_c, comment_sick])
        comment_c = pd.concat([comment_c, comment_holy])
        
        #This saves the partially cleansed comments to a file
        comment_df = comment_c
        comment_c.to_csv (r'sample_files\\user_comments_sample.csv', index = False, header=True)
        
    #This takes the rated comments and splits them into a positive/neutral data sets. 
    #It then uses NLTK tweet tokenizer to tokenize the comments of each user storing them into
    #a list of bag of words
    def createTokens(self):
        #This initializes NLTK's TweetTokenizer
        tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        
        #NB This uses the training data set from LoadComments
        v_positive_comments = self.comment_df[self.comment_df['sentiment'] == 4]
        positive_comments = self.comment_df[self.comment_df['sentiment'] == 3]
        neutral_comments = self.comment_df[self.comment_df['sentiment'] <= 2]
        
        self.v_positive_tokens = []
        self.positive_tokens = []
        self.neutral_tokens = []
        
        #This calls the tweet tokenizer for each "very positive" rated comment
        #and tokenize them
        for count in range(len(v_positive_comments)):
            self.v_positive_tokens.append(tknzr.tokenize(v_positive_comments.iloc[count]["body"]))
        
        #This calls the tweet tokenizer for each "positive" rated comment
        #and tokenize them
        for count in range(len(positive_comments)):            
            self.positive_tokens.append(tknzr.tokenize(positive_comments.iloc[count]["body"]))
        
        #This calls the tweet tokenizer for each "neutral/other" rated comment
        #and tokenize them
        for count in range(len(neutral_comments)):
            self.neutral_tokens.append(tknzr.tokenize(neutral_comments.iloc[count]["body"]))
    
    #This function lemmatizes tokens as well as do further cleaning of the data
    def cleanTokens(self, comment_tokens, stop_words = ()):
        cleaned_tokens = []
        
        #This initializes NLTK's WordNetLemmatizer object. This will be used to lemmatize
        #words. Eg/ It will turn 'walking', 'walks', 'walked' into 'walk'
        lemmatizer = WordNetLemmatizer()
        
        for token, tag in pos_tag(comment_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)
            
            #This looks at the tag of the word obtained from NLTK's TweetTokenizer and indicates
            #it as a nound, verb, or abjective respectively.
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            
            #This lemmatize the token
            token = lemmatizer.lemmatize(token, pos)
            
            #This does further cleaning based on an investigation of the nosleep Comments. These are
            #removed as they are just noise. Note that a check is also done against the stop words
            #passed to the function
            if len(token) > 0 and token != '...' and token != '…' and token != '’' and token != '‘' and token != '..' and token not in string.punctuation and token.lower() not in stop_words and token.lower() != "#x200b":
                cleaned_tokens.append(token.lower())
        
        return cleaned_tokens
    
    #This takes a list of tokens and converts them into a dictionary format that is needed
    #for NLTK's Naive Classifier model
    def convertTokensToDict(self, tokens_c):
        for tokens in tokens_c:
            yield dict([token, True] for token in tokens)    
    
    #This uses the v pos/pos/neu tokens create in createToken to clean them, converts
    #them into the format needed for the Naive Classifier model, create data sets
    #based on the classification, combine them, shuffle them, then finally creates
    #training / test data base on the training ratio defined in __Init__
    def createTrainingTestDataSets(self):
        #This loads NLTK's stop words
        stop_words = stopwords.words('english')
        
        positive_tokens_c = []        
        v_positive_tokens_c = []
        neutral_tokens_c = []
        
        #This cleans the very positive tokens by removing unwanted words, lemmatizing them, etc.
        for count in range(len(self.v_positive_tokens)):
            v_positive_tokens_c.append(self.cleanTokens(self.v_positive_tokens[count], stop_words))
        
        #This cleans the positive tokens by removing unwanted words, lemmatizing them, etc.
        for count in range(len(self.positive_tokens)):
            positive_tokens_c.append(self.cleanTokens(self.positive_tokens[count], stop_words))
        
        #This cleans the neutral tokens by removing unwanted words, lemmatizing them, etc.
        for count in range(len(self.neutral_tokens)):
            neutral_tokens_c.append(self.cleanTokens(self.neutral_tokens[count], stop_words))
            
        v_pos_tokens_mod = self.convertTokensToDict(v_positive_tokens_c)
        pos_tokens_mod = self.convertTokensToDict(positive_tokens_c)
        neu_tokens_mod = self.convertTokensToDict(neutral_tokens_c)            
    
        #This takes the positive/neutral dictionaries created above and put them into data sets.
        #It then merges them together, shuffle them and create train/test versions of the data sets
        #using 70/30% of the total data set respectively
        v_positive_dataset = [(comment_dict, "Very Positive")
                              for comment_dict in v_pos_tokens_mod]
        
        positive_dataset = [(comment_dict, "Positive")
                            for comment_dict in pos_tokens_mod]
        
        neutral_dataset = [(comment_dict, "Neutral")
                           for comment_dict in neu_tokens_mod]
        
        dataset = v_positive_dataset + positive_dataset + neutral_dataset 
        
        train_size = int(len(dataset)*self.train_ratio)
        random.shuffle(dataset)
        
        self.train_data = dataset[:train_size]
        self.test_data = dataset[train_size:]
        
    #This stores the train and test data sets created into JSON files. These
    #files can be used later on to load into the model without having to go through
    #the work of cleaning up the data etc again
    def saveModelData(self):
        with open('nltk_naive_class_model_data\model_train_nltk.json', 'w') as json_file:
            json.dump(self.train_data, json_file)
            
        with open('nltk_naive_class_model_data\model_test_nltk.json', 'w') as json_file:
            json.dump(self.test_data, json_file)
    
    #This loads the train and test data from the json files created into the
    #final objects to be loaded into the NaiveBayesClassifier model
    def loadModelData(self):
        #Note that this model data is based on the original version where 5 categories
        #were being applied - Very Negative, Negative, Neutral, Positive and Very Positive
        #The current application of the model is to treat Very Negative/Negative as Neutral
        #as well under the premise that the commenters are engaging with the story through
        #'role playing'
        with open('nltk_naive_class_model_data\model_train_nltk.json') as json_file:
            self.train_data = json.load(json_file)
        
        with open('nltk_naive_class_model_data\model_test_nltk.json') as json_file:
            self.test_data = json.load(json_file)
    
    def generateModel(self):
        #This creates a NaiveBayesClassifier model based on the train_data created
        #above then it tests it using the test_data set. Ideally, we want the accuracy
        #to be as high as possible. Right now it is about 55%
        self.classifier = NaiveBayesClassifier.train(self.train_data)
        
        #Uncomment if you want to see what the Accuracy and the top 10 words of the model
        #is. Note you may choose to return this information based on your implementation        
        print("Accuracy is:", classify.accuracy(self.classifier, self.test_data))
        print(self.classifier.show_most_informative_features(10))
    
    #This uses the model created in generateModel to predict/rate the comments provided
    #in the test data set. Note this comprise of data the model hasn't seen. This is done
    #using both the NLTK model and the Vader model and an average of the score is currently
    #being used as the prediction. The goal in the future is to only use the NLTK model
    #once it becomes more robust using techniques such as bigrams etc.
    #The sentiment column can be compared against the prediction model to gauge accuracy, etc
    def predictTestData(self):
        #This initializes NLTK's TweetTokenizer
        tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        
        #This initializes Vader Sentiment Analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        predictions_nltk = []
        predictions_vader = []
        predictions = []
        
        for count in range(self.test_data_records):
            #This creates cleansed tokens on the current comment and uses the NLTK's naive classifier
            #model to predict what type of comment it is
            custom_tokens = self.cleanTokens(tknzr.tokenize(self.random_df.iloc[count]["body"]))
            predicted_sentiment = self.classifier.classify(dict([token, True] for token in custom_tokens))
            
            #This assigns a score to the prediction based on the type of comment predicted
            #For now, we are only using Positive, Very Positive and Neutral (Other)
            if predicted_sentiment == "Very Positive":
                predictions_nltk.append(4)
                
            if predicted_sentiment == "Positive":
                predictions_nltk.append(3)
                
            if predicted_sentiment == "Neutral" or predicted_sentiment == "Negative" or predicted_sentiment == "Very Negative":
                predictions_nltk.append(2)
            
            #This uses the vader sentiment analyzer to calculate a compound score. This is a value
            #between -1 and 1 that indicates how negative or positive the comment is.
            vader_compound_score = analyzer.polarity_scores(self.random_df.iloc[count]["body"])
            
            #This assigns a score to the prediction based on the type of comment predicted
            #For now, we are only using Positive, Very Positive and Neutral (Other)
            if vader_compound_score["compound"] >= 0.5:
                predictions_vader.append(4)
            
            if vader_compound_score["compound"] >= 0 and vader_compound_score["compound"] < 0.5:
                predictions_vader.append(3)
                
            if vader_compound_score["compound"] < 0:
                predictions_vader.append(2)
            
            #This calculates the final prediction (average of both NLTK and vader)
            predictions.append(math.ceil((predictions_vader[count] + predictions_nltk[count]) / 2))
        
        self.random_df['prediction_vader'] = predictions_vader
        self.random_df['prediction_nltk'] = predictions_nltk
        self.random_df['prediction'] = predictions
        
        self.random_df.to_csv (r'sample_files\\user_comments_model_val_sample.csv', index = False, header=True)

    #This uses the final model created in generateModel to predict/rate the comments provided
    #in the full data set. This is done using both the NLTK model and the Vader model and an 
    #average of the score is currently being used as the prediction. The goal in the future is 
    #to only use the NLTK model once it becomes more robust using techniques such as bigrams etc.
    #The sentiment column can be compared against the prediction model to gauge accuracy, etc
    def predictData(self):
        #This initializes NLTK's TweetTokenizer
        tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        
        #This initializes Vader Sentiment Analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        predictions_nltk = []
        predictions_vader = []
        predictions = []
        
        for count in range(len(self.comment_full)):
            #This creates cleansed tokens on the current comment and uses the NLTK's naive classifier
            #model to predict what type of comment it is
            custom_tokens = self.cleanTokens(tknzr.tokenize(self.comment_full.iloc[count]["body"]))
            predicted_sentiment = self.classifier.classify(dict([token, True] for token in custom_tokens))
            
            #This assigns a score to the prediction based on the type of comment predicted
            #For now, we are only using Positive, Very Positive and Neutral (Other)
            if predicted_sentiment == "Very Positive":
                predictions_nltk.append(4)
                
            if predicted_sentiment == "Positive":
                predictions_nltk.append(3)
                
            if predicted_sentiment == "Neutral" or predicted_sentiment == "Negative" or predicted_sentiment == "Very Negative":
                predictions_nltk.append(2)
            
            #This uses the vader sentiment analyzer to calculate a compound score. This is a value
            #between -1 and 1 that indicates how negative or positive the comment is.
            vader_compound_score = analyzer.polarity_scores(self.comment_full.iloc[count]["body"])
            
            #This assigns a score to the prediction based on the type of comment predicted
            #For now, we are only using Positive, Very Positive and Neutral (Other)
            if vader_compound_score["compound"] >= 0.5:
                predictions_vader.append(4)
            
            if vader_compound_score["compound"] >= 0 and vader_compound_score["compound"] < 0.5:
                predictions_vader.append(3)
                
            if vader_compound_score["compound"] < 0:
                predictions_vader.append(2)
            
            #This calculates the final prediction (average of both NLTK and vader)
            predictions.append(math.ceil((predictions_vader[count] + predictions_nltk[count]) / 2))
        
        self.comment_full['prediction_vader'] = predictions_vader
        self.comment_full['prediction_nltk'] = predictions_nltk
        self.comment_full['prediction'] = predictions
        
        self.comment_full.to_csv (r'sample_files\\user_comments_final_sample.csv', index = False, header=True)
    
    #This function loads the distance dictionary file generated from the site Topic Modelling models
    #Note the file must have a structure like the following:
    #{"abf59b": ["aopi3m"], "bzf59b": ["aopi3m"], ... }
    #Each story could have 1 or more recommended stories
    def loadDistanceDict(self):
        #This sample file shows the expected structure. Note for this project, it actually consisted of about 47k
        #stories with about 200 recommended stories each. The file was about 100MB.
        file = open("sample_files\\distance_dict_sample.txt", "r")
        recommended_data = file.read()
        self.stories_dict = ast.literal_eval(recommended_data)
        file.close()
    
    #This function ranks each recommended story suggests by the site's Topic Modelling models
    #by applying the following formula to it
    #Score = 4^(No. of Commenters commenting on both stories) - 2 Log2(Commenters' Total Commenters)
    #        + Story Total Comments
    #This is dependent on all of the comments for the stories already being given a rating based on
    #the NLTK Naive Classifier/Vader Models built.
    #NB/ If a recommended story has no comments, then it is excluded from the final ranked list
    def calcScores(self):
        ranking_dict = {}
        scores_dict = {}
    
        #This loads the distance dictionary file as specified above in the function definition
        loadDistinceDict()
        
        #This loads the comment files that have the generated predictions. Note this could be changed
        #to read from a dynamodb table etc
        score_df = pd.read_csv('sample_files\\user_comments_sample.csv')
        
        #This has the total comments each user madde for convenience. Note this could be derived from
        #the user_comments file by using a group by on the author
        commenters_df = pd.read_csv('sample_files\\total_comments_sample.csv')        
        
        #This changes the prediction into a float and divide by 2 thus each story will have an initial
        #prediction between 1 and 2
        score_df['score'] = score_df['prediction'].astype(float) / 2
        
        #This penalizes the score by taking the log2 of the total comments each commenter made. This is
        #using an assumption that a super commenter may not have a preference over someone who only
        #comments on specific types of stories
        commenters_df['total_comments'] = np.log(commenters_df['total_comments']) * -2
        stories_df = score_df.groupby("storyId")["score"].count().reset_index()
        
        #This gets the unique number of authors/commenters to be used in score_df_B
        score_df_B = score_df.groupby(['storyId', 'author'])["score"].sum().reset_index().rename(columns={'score':'score_B'}) 
        
        #This does a join against itself to see how many authors are commenting on different stories        
        #This is then assign to score_df_C
        score_df_C = score_df_B.copy()
        score_df_C = pd.merge(score_df_C, score_df_C, on='author', how='inner')
        
        #This assigns the commenters total comments to score_df_B
        score_df_B = pd.merge(score_df_B, commenters_df, on='author', how='inner')
        score_df_B = score_df_B.groupby("storyId")["total_comments"].sum().reset_index().rename(columns={'total_comments':'score_B','storyId':'storyId_B'})
        
        #This groups the stories summing up the individual prediction scores then assign it to score_df_A
        score_df_A = score_df.groupby("storyId")["score"].sum().reset_index().rename(columns={'score':'score_A','storyId':'storyId_A'})
        
        #This keeps track of the total recommendations made for each story so that an average can be calculated
        count = 1
        total_recommendations = 0
        recommendations_found = 0
        
        #This loops through all of the story ids found in the distance dictionary file
        for storyId in self.stories_dict.keys():
            stories_D_dict = {"storyId_D": []
                             }        
            
            #This takes the list of recommended stories for each story id in the distance dictionary
            #file and put it in a dictionary object
            for count2 in range(len(self.stories_dict[storyId])):
                stories_D_dict["storyId_D"].append(self.stories_dict[storyId][count2])
            
            #This creates a list of recommended stories in a new data set score_df_D
            #This data set is used to filter out the results
            score_df_D = pd.DataFrame(stories_D_dict)
            
            #This takes a copy of score_df_C and filter out everything that doesn't have to do
            #with the current story id being worked on. Basically, it is looking to see if any
            #commenter has commented on both the current story id and each of its recommended story ids
            score_df_C_C = score_df_C.copy()
            score_df_C_C = score_df_C_C[score_df_C_C['storyId_y'] == storyId]
            score_df_C_C = score_df_C_C[score_df_C_C['storyId_x'] != storyId]
            
            score_df_C_C = score_df_C_C.groupby("storyId_x")["score_B_x"].count().reset_index().rename(columns={'score_B_x':'score_C','storyId_x':'storyId_C'})
            score_df_C_C['score_C'] = np.power(4, score_df_C_C['score_C'])
            
            #This uses a new data set score_df_F by doing a join on all the other data sets.
            #This allows us to have the data merged together which can then be used to calculate the score
            score_df_F = pd.merge(score_df_A, score_df_D, left_on='storyId_A', right_on='storyId_D', how='inner')
            score_df_F = pd.merge(score_df_F, score_df_B, left_on='storyId_A', right_on='storyId_B', how='left')
            score_df_F = pd.merge(score_df_F, score_df_C_C, left_on='storyId_A', right_on='storyId_C', how='left')
            
            score_df_F['score_C'] = score_df_F['score_C'].fillna(0)
            score_df_F['score_B'] = score_df_F['score_B'].fillna(0)
            score_df_F['score'] = score_df_F['score_A'] + score_df_F['score_B'] + score_df_F['score_C']
            
            #This sorts the score for the recommended story ids in descending order and also remove
            #the score for the current story id itself
            score_df_F = score_df_F.sort_values(by='score', ascending=False)
            score_df_F = score_df_F[score_df_F['storyId_A'] != storyId]
            rankings = []
            scores = []
            
            #If the len of the new data set is 0, that means that none of the recommended stories
            #has any comments thus the passed recommended story ids are used as the ranked list
            if len(score_df_F) < 1:
                rankings = self.stories_dict[storyId]
                scores = [0] * len(self.stories_dict[storyId])
            else:
                #This implies at least 1 story has comments. Here, the number of ranked recommended
                #stories found is added to the total recommended list.
                total_recommendations += len(score_df_F)
                recommendations_found += 1
                
                #This adds the recommended ranked story ids and their scores into lists
                for count2 in range(len(score_df_F)):
                    rankings.append(score_df_F.iloc[count2]["storyId_A"])
                    scores.append(str(score_df_F.iloc[count2]["score"]))
            
            #This lists are then added to a dictionary for the current story id
            ranking_dict[storyId] = rankings
            scores_dict[storyId] = scores
            
            count += 1
                
        print('average recommendations per story - ' + str(total_recommendations / recommendations_found))
        
        #The dictionary showing the ranked story ids for each story is saved in the story_rankings.txt file
        f = open("story_rankings.txt","w")
        f.write( str(ranking_dict) )
        f.close()
        
        #The dictionary showing the scores of the ranked story ids for each story is saved in the story_rankings.txt file
        f = open("story_rankings_scores.txt","w")
        f.write( str(scores_dict) )
        f.close()        
        
    #This is an example showing the list of steps of taking precleaned data (typically you will want
    #to call the preCleaning function first for new Comments downloaded), loading it in the class
    #converting them to tokens, cleaning them, creating Training/Test Data Sets, creating the model
    #using the training data and finally predicting the Test Data and saving the results in a test
    #file for review
    def runExample(self):
        self.loadComments()
        self.createTokens()
        self.createTrainingTestDataSets()
        self.generateModel()
        self.predictTestData()
        self.predictData()