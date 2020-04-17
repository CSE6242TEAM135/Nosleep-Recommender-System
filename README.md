# Reddit/nosleep Recommender System

## The application is web based and assumes AWS infrastructure. 
Aws components used
•	S3 <br/>
•	DynamoDB<br/>
•	EC 2 (Virtual Machine – RedHat)<br/>
## Following libraries need to be installed on the RedHat machine
•	Python 3.7.4 <br/>
•	Django <br/>
•	Pandas <br/>
•	Boto3 <br/>
•	NLTK <br/>
•	Wordcloud <br/>
•	Plotly <br/>

### To install any Python library use this syntax : python3 -m  pip install --user plotly
In addition, AWS CLI must be installed and configured with appropriate Access keys , which will allow to communicate with DynamoDB

### To install the software, <br/> log into EC2 and run this command
git clone https://github.com/CSE6242TEAM135/Nosleep-Recommender-System.git .
This will pull all the required files.
Then type this command:<br/> python3 Nosleep-Recommender-System/NoSleepRecommender_DJANGO/manage.py runserver 0.0.0.0:8000 & <br/>
It will start the server. Thereafter do <br/> ctrl+a+d<br/> This will continue running the server in the background and you can safely exit the CLI

## Structure of Git:
<b>Model</b> folder contains our machine learning models, which includes Topic Modeling and scoring methodology
<b>NoSleepRecommender_DJANGO</b> folder contains Django web server, Wordcloud  and Network graph files

