
# Reddit/nosleep Recommender System
## Working software can be viewed at 
 http://ec2-54-86-31-115.compute-1.amazonaws.com:8000/ <br/>
----------------------
### Important
To run a search you would need story IDs. The rationale is that if we need to hook this up with Reddit, the parameter passed to our system will be a story ID, which is unique. We have huge dataset of story IDs ran between 01/01/2019 and 07/01/2019 (Six months) <br/>
We are providing some story IDs, which you can use to test:<br/>
abg1pv , abg4dj, abg8cw, abgcly,  abgd7w, abgfyo, abgjjn, abgyue, abgzs9, abhcls, abhgjs <br/>

PS: Complete list of story IDs can be found in storyids.txt here:
https://github.com/CSE6242TEAM135/Nosleep-Recommender-System/blob/master/storyids.txt


--------------------------------------------
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
•	Networkx <br/>

### To install any Python library use this syntax : python3 -m  pip install --user plotly
In addition, AWS CLI must be installed and configured with appropriate Access keys , which will allow to communicate with DynamoDB

### To install our software, <br/> 
Log into EC2 and run these commands:<br/>
• git clone https://github.com/CSE6242TEAM135/Nosleep-Recommender-System.git <br/>
  This will pull all the required files.<br/>
• Then type this command:<br/> python3 Nosleep-Recommender-System/NoSleepRecommender_DJANGO/manage.py runserver 0.0.0.0:8000 & <br/>
It will start the server. <br/>
• Thereafter do <br/> ctrl+a+d<br/> This will continue running the server in the background and you can safely exit the CLI

## Structure of Git:
<b>Model</b> folder contains our machine learning models, which includes Topic Modeling and scoring methodology <br/>
<b>NoSleepRecommender_DJANGO</b> folder contains Django web server, Wordcloud  and Network graph files <br/>
<b> storyids.txt </b> contains list of complete story ids that can be fetched from AWS.

