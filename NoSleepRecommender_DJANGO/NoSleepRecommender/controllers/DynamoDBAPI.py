import boto3
import pandas as pd

class DynamoDBAPI:
    def __init__(self):
        #This creates the dynamoDB object that points to the location of the database
        #Note this requires the AWSCLI connection details to be setup
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1', endpoint_url="https://dynamodb.us-east-1.amazonaws.com")


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

        table = self.dynamodb.Table('CommentsNoSleep')

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

        table = self.dynamodb.Table('StoriesNoSleep')

        response = table.scan()

        for item in response['Items']:
            #print(item['title'])
            if 'title' in item:
                stories_dict["title"].append(item['title'])
                stories_dict["body"].append(item['selftext'])
                stories_dict["author"].append(item['author'])
                stories_dict["story_id"].append(item['id'])

        story_df = pd.DataFrame(stories_dict)
        story_df = story_df.head(10)
        return story_df