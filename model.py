# imports
from pytz import timezone
import time
import math
from datetime import datetime
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline


# logging 
last_time = time.time()
def log(*args):
    """
        Function to log model processing
    """

    # elapsed time 
    global last_time 

    # log 
    print("\x1b[2m", datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S'),
          "\x1b[1m\x1b[97m", "[MODEL] ", "\x1b[0m", ''.join(args), "\x1b[1m\x1b[97m | \x1b[91m", math.floor(time.time() - last_time),end="s\x1b[0m\n")
    last_time = time.time()



# reading data
log("Processing Data....")
df1 = pd.read_csv('./data/dialogs1.csv', sep='\t')
df2 = pd.read_csv('./data/dialogs2.csv', sep=',')

# merge
DF = pd.merge(df1, df2)
DF = DF[['question', 'answer']]
log("Data Processed....")

# model classifiers
log("Started Model Processing....")
MODEL = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])

# data fit
MODEL.fit(DF['question'], DF['answer'])
log("Model Processing Done...")


# demo output 
while True: 
    user = input("Chat >")
    output = MODEL.predict([user])[0]
    print(output)