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

    # result is an array with the popularity score
    # e.g. array([10.08465046])
    # so we extract the float to send in a dict with the title

    pred = results[0]

    return {"title": str(title), "prediction": float(pred)}


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

if __name__ == "__main__":
    """
    /predict?
    title=Harry%20Potter&
    original_title=Harry%20Potter&
    release_date=2010-06-09&
    duration_min=150&
    description=Harry%20is%20a%20wizard%20that%20tries%20to%20save%20the%20world%20from%20crazy%20guys&
    budget=1000000&
    original_language=en&
    status=Released&
    number_of_awards_won=80&
    number_of_nominations=120&
    has_collection=1&
    all_genres=Fantasy,%20Family,%20Adventure&
    top_countries=United%20States%20of%20America,,%20United%20Kindgom&
    number_of_top_productions=3&
    available_in_english=True
    """

    original_title = "test-original_title"
    title = "test-title"
    duration_min = 70
    release_date = 1978
    description = "test-description"
    budget = 170009
    original_language = "test-original_language"
    status = 0
    number_of_awards_won = 10
    number_of_nominations = 50
    has_collection = 1
    all_genres = "test-all_genres"
    top_countries = "test-top_countries"
    number_of_top_productions = 124
    available_in_english = True

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

    print(X.dtypes)
    print(X)
