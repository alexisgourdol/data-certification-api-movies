import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# define a root `/` endpoint
@app.get("/")
def index():
    return {"okay": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(
    original_title,
    title,
    release_date,
    duration_min,
    description,
    budget,
    original_language,
    status,
    number_of_awards_won,
    number_of_nominations,
    has_collection,
    all_genres,
    top_countries,
    number_of_top_productions,
    available_in_english,
):

    # create dataframe for prediction
    X = pd.DataFrame(
        dict(
            original_title=[str(original_title)],
            title=[str(title)],
            release_date=[str(release_date)],
            duration_min=[float(duration_min)],
            description=[str(description)],
            budget=[float(budget)],
            original_language=[str(original_language)],
            status=[str(status)],
            number_of_awards_won=[int(number_of_awards_won)],
            number_of_nominations=[int(number_of_nominations)],
            has_collection=[int(has_collection)],
            all_genres=[str(all_genres)],
            top_countries=[str(top_countries)],
            number_of_top_productions=[float(number_of_top_productions)],
            available_in_english=[bool(available_in_english)],
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

    return dict(prediction=pred)"""


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

if __name__== "__main__":
