# imports
import time
import math
from datetime import datetime
# import pickle
from pytz import timezone
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
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
DF = pd.read_csv('./data/dialogs.txt', sep='|')
log("Data Processed....")

# model classifiers
log("Started Model Processing....")
MODEL = Pipeline([
    ('bow', CountVectorizer(max_df=0.6, min_df=5)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=0))
])

# data fit
MODEL.fit(DF['question'], DF['answer'])
log("Model Processing Done...")


# demo output
while True:
    user = input("Chat > ")
    output = MODEL.predict([user])[0]
    print(output)
