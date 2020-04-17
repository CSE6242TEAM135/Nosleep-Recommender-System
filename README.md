# Nosleep-Recommender-System
Reddit/nosleep Recommender System
The application use web based and assumes AWS infrastructure.
## Aws components used
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

### To install the software use this syntax : python3 -m  pip install --user plotly
In addition, AWS CLI must be installed and configured with appropriate Access keys , which will allow to communicate to DynamoDB

### To install the software, <br/> log into EC2 and run this command
git clone https://github.com/CSE6242TEAM135/Nosleep-Recommender-System.git .
This will pull all the required files.
Then type in this command:<br/> python3 Nosleep-Recommender-System/NoSleepRecommender_DJANGO/manage.py runserver 0.0.0.0:8000 &
It will start the server. Thereafter do <br/> ctrl+a+d<br/>. This will continue running the server in the background and you can safely exit.

## Structure of Git:
Model folder contains our machine learning models, which includes Topic Modeling and scoring methodology
NoSleepRecommender_DJANGO folder contains Django web server, Wordcloud  and Network graph files

