# standard imports
import math
import pickle
import time
from datetime import datetime

# module imports
import pandas as pd
from pytz import timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

# logging
last_time = time.time()


def log(*args):
    """Do logs for model processing."""

    global last_time
    # log
    print(
        "\x1b[2m",
        datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S"),
        "\x1b[1m\x1b[97m",
        "[MODEL] ",
        "\x1b[0m",
        "".join(args),
        "\x1b[1m\x1b[97m | \x1b[91m",
        math.floor(time.time() - last_time),
        end="s\x1b[0m\n",
    )
    last_time = time.time()


# reading data
log("Processing Data....")
DF = pd.read_csv("./data/dialogs.txt", sep="|")
log("Data Processed....")

# model classifiers
log("Started Model Processing....")
MODEL = Pipeline(
    [
        ("bow", CountVectorizer()),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
        ("classifier", RandomForestClassifier(n_estimators=100)),
    ]
)

# data fit
MODEL.fit(DF["question"], DF["answer"])
log("Model Processing Done...")

# dump
pickle.dump(MODEL, open("model.pkl", "wb"))
