from django.shortcuts import render
from NoSleepRecommender.controllers.RedditAPI import RedditAPI
from NoSleepRecommender.controllers.DynamoDBAPI import DynamoDBAPI
from NoSleepRecommender.plotly_wordcloud import plotly_wordcloud


def home(request):
    args = {}
    text = ""
    args['mytext'] = text
    db = DynamoDBAPI()
    args['recent_stories'] = db.get_stories()
    return render(request, 'index.html', args)


def results(request):
    args = {}
    text = ""
    stories = {}

    db = DynamoDBAPI()

    if request.method == 'POST':
        text = request.POST.get('search_key')
        stories = db.get_recommendations(text)
    text = ''
    args['results'] = text
    args['current_story'] = stories['current_story']
    args['recommendations'] = stories['recommendations']
    args['total_recommendations'] = len(stories['recommendations'])
    for story in stories['recommendations'].body:
        text += story
    args['wordcloud'] = plotly_wordcloud(text)

    return render(request, 'results.html', args)


def charts(request):
    args = {}
    return render(request, 'charts.html', args)
