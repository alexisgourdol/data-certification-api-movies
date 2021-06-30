import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(
    acousticness,
    danceability,
    duration_ms,
    energy,
    explicit,
    id,
    instrumentalness,
    key,
    liveness,
    loudness,
    mode,
    name,
    release_date,
    speechiness,
    tempo,
    valence,
    artist,
):

    # create dataframe for prediction
    X = pd.DataFrame(
        dict(
            key=[key],
            pickup_datetime=[formatted_pickup_datetime],
            pickup_longitude=[float(pickup_longitude)],
            pickup_latitude=[float(pickup_latitude)],
            dropoff_longitude=[float(dropoff_longitude)],
            dropoff_latitude=[float(dropoff_latitude)],
            passenger_count=[int(passenger_count)],
        )
    )

    # load local pipe
    pipeline = joblib.load("model.joblib")

    # make prediction of popularity
    results = pipeline.predict(X)
    print(type(results))

    # convert response from numpy to python type
    # pred = float(results[0])
    pred = results
    print(type(pred))

    return dict(prediction=pred)


"""the pipeline expects to be trained with a DataFrame containing
the following data types in that order
```
original_title              string
title                       string
release_date                string
duration_min                float
description                 string
budget                      float
original_language           string
status                      string
number_of_awards_won        int
number_of_nominations       int
has_collection              int
all_genres                  string
top_countries               string
number_of_top_productions   float
available_in_english        bool"""

"""
acousticness,
danceability,
duration_ms,
energy,
explicit,
id,
instrumentalness,
key,
liveness,
loudness,
mode,
name,
release_date,
speechiness,
tempo,
valence,
artist,
"""
