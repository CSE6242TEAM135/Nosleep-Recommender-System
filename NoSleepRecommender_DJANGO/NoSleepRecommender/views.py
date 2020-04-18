from django.shortcuts import render
from NoSleepRecommender.controllers.DynamoDBAPI import DynamoDBAPI
from NoSleepRecommender.controllers.plotly_wordcloud import plotly_wordcloud
from NoSleepRecommender.controllers.NetworkGraph import plotly_NetworkGraph

def home(request):
    args = {}
    args['no_results'] = False
    db = DynamoDBAPI()
    args['recent_stories'] = db.get_stories()
    return render(request, 'index.html', args)


def results(request):
    args = {}
    db = DynamoDBAPI()

    if request.method == 'POST':
        text = request.POST.get('search_key')
        stories = db.get_recommendations(text)
        args['results'] = text
        args['current_story'] = stories['current_story']
        args['recommendations'] = stories['recommendations']
        args['total_recommendations'] = len(stories['recommendations'])
        if args['total_recommendations'] > 0:
            for story in stories['recommendations'].body:
                text += story
            args['wordcloud'] = plotly_wordcloud(text)
            #print(stories['ranked_list'])
            args['networkGraph'] = plotly_NetworkGraph(stories)

    if args['total_recommendations'] == 0:
        args['no_results'] = True
        return render(request, 'index.html', {})

    return render(request, 'results.html', args)


def contact(request):
    args = {}
    return render(request, 'contact.html', args)
