import pandas as pd
import praw

class RedditAPI:
    def __init__(self):
        PERSONAL_USE_SCRIPT_14_CHARS = 'DlOSzt8yX1sHEA'
        SECRET_KEY_27_CHARS = 'akhRmZhbpUVRlPOc67CeJQSqIDY'
        YOUR_APP_NAME = 'NoSleepRecommender'
        YOUR_REDDIT_USER_NAME = 'josephs7'
        YOUR_REDDIT_LOGIN_PASSWORD = 'Computer1'
        self.reddit = praw.Reddit(client_id=PERSONAL_USE_SCRIPT_14_CHARS,
                             client_secret=SECRET_KEY_27_CHARS,
                             password=YOUR_REDDIT_LOGIN_PASSWORD,
                             user_agent=YOUR_APP_NAME,
                             username=YOUR_REDDIT_USER_NAME)
        self.subreddit = self.reddit.subreddit('nosleep')

    def get_stories(self):
        stories_dict = {"story_id": [],
                        "title": [],
                       "author": [],
                       "body": []}

        my_subreddit = self.subreddit.hot(limit=10)
        for submission in my_subreddit:
            stories_dict["title"].append(submission.title)
            stories_dict["body"].append(submission.selftext)
            stories_dict["author"].append(submission.author)
            stories_dict["story_id"].append(submission.id)

        story_df = pd.DataFrame(stories_dict)
        story_df = story_df.head(10)
        story_df = story_df.drop([0, 1], )
        return story_df

