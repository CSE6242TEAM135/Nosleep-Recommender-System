import boto3
from boto3.dynamodb.conditions import Key
import pandas as pd
from sklearn.preprocessing import minmax_scale
import math

class DynamoDBAPI:
    def __init__(self):
        #This creates the dynamoDB object that points to the location of the database
        #Note this requires the AWSCLI connection details to be setup
        self.resource = boto3.resource('dynamodb', region_name='us-east-1', endpoint_url="https://dynamodb.us-east-1.amazonaws.com")

    def get_comments(self):
        comments_dict = {"link_id": [],
                         "sortKey": [],
                         "score": [],
                         "permalink": [],
                         "author_fullname": [],
                         "id": [],
                         "storyId": [],
                         "author": [],
                         "parent_id": [],
                         "body": []
                         }

        table = self.resource.Table('CommentsNoSleep')

        itemsList = []
        response = table.scan()

        print(response)

        for i in response['Items']:
            for key in i.keys():
                comments_dict[key].append(i[key])

        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])

            for i in response['Items']:
                for key in i.keys():
                    comments_dict[key].append(i[key])

        df = pd.DataFrame(comments_dict)
        df = df.head(10)
        return df

    def get_stories(self):
        stories_dict = {"story_id": [],
                        "title": [],
                       "author": [],
                       "body": []}

        table = self.resource.Table('StoriesNoSleep')

        response = table.scan()
        for item in response['Items']:
            if 'title' in item:
                stories_dict["title"].append(item['title'])
                stories_dict["body"].append(item['selftext'])
                stories_dict["author"].append(item['author'])
                stories_dict["story_id"].append(item['id'])

        story_df = pd.DataFrame(stories_dict)
        story_df = story_df.head(10)
        return story_df

    def get_recommendations(self, story_id):
        recommended_stories_dict = {"story_id": [],
                                    "title": [],
                                    "author": [],
                                    "body": [],
                                    "recommendations": [],
                                    "score": []}

        current_story = self.get_story_by_id(story_id)

        if len(current_story) == 0:
            return {"current_story": {}, "recommendations": {}}

        recommendations = eval(str(current_story["recommendations"]))
        scores = eval(str(current_story["recommendations_scores"]))

        ranking_list = recommendations
        scores_list = scores
        scores_num = [float(i) for i in scores_list]
        scores_list_std = minmax_scale(scores_num)
        ranked_list = {ranking_list[i]: {'score': scores_list_std[i]} for i in range(10)}

        for i in ranked_list:
            counter = 0
            for item in recommendations:
                if i == item:
                    #print(i + " == " + item)
                    item_story = self.get_story_by_id(item)
                    if 'title' in item_story and 'body' in item_story:
                        recommended_stories_dict["title"].append(item_story['title'])
                        recommended_stories_dict["body"].append(item_story['body'])
                        recommended_stories_dict["author"].append(item_story['author'])
                        recommended_stories_dict["story_id"].append(item_story['id'])
                        recommended_stories_dict["recommendations"].append(item_story['recommendations'])
                        recommended_stories_dict["score"].append(scores[counter])
                counter += 1
        recommendations = pd.DataFrame(recommended_stories_dict)

        return {"current_story": current_story, "recommendations": recommendations, "ranked_list": ranked_list}


    def get_story_by_id(self, story_id):
        result = {}
        resp = self.query_table(
            table_name='StoriesNoSleep',
            key='id',
            value=story_id
        )
        items = resp.get('Items')

        if len(items) > 0:
            result["title"] = items[0]['title']
            result["body"] = items[0]['selftext']
            result["author"] = items[0]['author']
            result["id"] = items[0]['id']
            result["recommendations"] = items[0]['recommended_storyIds']
            result["recommendations_scores"] = items[0]['recommended_scores']

        return result

    def query_table(self, table_name, key=None, value=None):
        table = self.resource.Table(table_name)

        if key is not None and value is not None:
            filtering_exp = Key(key).eq(value)
            return table.query(KeyConditionExpression=filtering_exp)

        raise ValueError('Parameters missing or invalid')
