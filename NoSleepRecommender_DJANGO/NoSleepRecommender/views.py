from django.shortcuts import render
from NoSleepRecommender.controllers.RedditAPI import RedditAPI
from NoSleepRecommender.controllers.DynamoDBAPI import DynamoDBAPI

def home(request):
    args = {}
    text = ""
    args['mytext'] = text
    rd = RedditAPI()
    stories = rd.get_stories()
    #print(stories)
    args['recent_stories'] = stories
    return render(request, 'index.html', args)


def results(request):
    args = {}
    text = ""

    db = DynamoDBAPI()
    stories = db.get_stories()
    print(stories)
    if request.method == 'POST':
        text = request.POST.get('search_key')
        print(request.POST.get('search_key'))
    args['results'] = text
    args['recommendations'] = stories
    return render(request, 'results.html', args)
