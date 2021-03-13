Assignment 2 ADSI
==============================

This repo is the project code for a beer style prediction API. This API can be found at https://afternoon-ocean-26363.herokuapp.com/

The purpose of this project is to accurately predict the beer style of a particular beer based on a BeerAdvocates users’ rating of the beer. Fields include criteria such as appearance, aroma, palate or taste, as well as the name of hte brewery.

Available Endpoints
------------

* `/` GET: brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo. No request body or parameters required.
* `/health/` GET: a standard 'hello world' response. No request body or parameters required.
* `/beer/type/` POST: returns a prediction for a single input only. Example request body:
```json
{
  "review_overall": 1,
  "review_aroma": 1,
  "review_appearance": 1,
  "review_palate": 1,
  "review_taste": 1,
  "brewery_names": "New England Brewing Co."
}
```
Example Response:
```json
{"class_name": "American Adjunct Lager"}
```
* `beers/types/` POST: Returns predictions for multiple inputs. Example request body"
```json
{
  "review_overall": [1,2],
  "review_aroma": [1,2],
  "review_appearance": [1,2],
  "review_palate": [3,2],
  "review_taste": [4,2],
  "brewery_names": ["New England Brewing Co.", "all strings accepted"]
}
```
Example response:
```json
{"class_name": ["American IPA", "American Adjunct Lager"]}
```

Getting Started
------------
This project runs on Python 3.7. It has not been tested on any other version of python. To launch the API ensure docker is installed.

```bash
docker build -t assignment_2 .
docker run -dit --name assignment_run -p 80:80 assignment_2
```



Deployments
------------


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
