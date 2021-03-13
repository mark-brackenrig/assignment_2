from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import torch
import pickle as pkl
from pydantic import BaseModel
from typing import List
from src.models.predict_model import create_beers_dataset, create_beer_dataset, predict
from src.models.pytorch import PytorchMultiClass,get_device

ohe = pkl.load(open('models/ohe.pkl','rb'))
scaler = pkl.load(open('models/scaler.pkl','rb'))
y_encoder = pkl.load(open('models/y_encoder.pkl','rb'))

model = PytorchMultiClass(num_features = 2504,class_num= 104)
model.load_state_dict(torch.load('models/beer_style_prediction.pt'))

app = FastAPI()

@app.get('/')
def display_project_description():
    """
        Displaying a brief description of the project objectives, list of endpoints, expected input parameters  and output format of the model, link to the Github repo related to this project
    """
        
    return {'Hello World Placeholder'}

@app.get('/health/')
def hello_world():
    """
    Health check for the API to determine that the API is running. Expected response is 'Hello world! App running.'

    Parameters
    ----------
    None

    Responses
    ---------
    200: Hello World! App running.

    """
    print("Hello World")
    return {"Hello World! App running."}
   
class beerType(BaseModel):
    """
    The request model for the API to return a single prediciton
    """
    review_aroma: float
    review_appearance: float
    review_palate: float
    review_taste: float
    brewery_names: str
    class Config:
        schema_extra = {
            'example': {
            "review_aroma": 5,
            "review_appearance": 4,
            "review_palate": 3,
            "review_taste": 5,
            "brewery_names": "Vecchio Birraio"
            }
        }
class beerTypeOut(BaseModel):
    class_name: str

    class Config:
        schema_extra = {
            "example":{
        "class_name": "American IPA"}
        }

@app.post('/beer/type/', response_model = beerTypeOut)
def predict_beer_type(request_body: beerType):
    """
    Returns a single beer style prediction.

    Parameters
    ----------
    * **review_overall** (float): Overall score given by reviewer - ranges from 1 to 5.
    * **review_aroma** (float): Score given by reviewer regarding beer aroma- ranges from 1 to 5.
    * **review_appearance** (float): Score given by reviewer regarding beer appearance - ranges from 1 to 5.
    * **review_palate** (float): Score given by reviewer regarding beer palate- ranges from 1 to 5.
    * **review_taste** (float): Score given by reviewer regarding beer taste - ranges from 1 to 5.
    * **brwery_name** (string): Name of brewery - ranges from 1 to 5.
    """
    data = create_beer_dataset(request_body,ohe,scaler)
    device = get_device()
    prediction = predict(data,model, device,y_encoder)
    print(prediction)
    return prediction

class beersType(BaseModel):
    """
    The request model for the API to return a single prediciton
    """
    review_aroma: List[float]
    review_appearance: List[float]
    review_palate: List[float]
    review_taste: List[float]
    brewery_names: List[str]
    class Config:
        schema_extra = {
            'example': {
            "review_aroma": [5],
            "review_appearance": [4],
            "review_palate": [3],
            "review_taste": [5],
            "brewery_names": ["Vecchio Birraio"]
            }
        }
class beersTypeOut(BaseModel):
    class_name: list

    class Config:
        schema_extra = {
            "example":{
        "class_name": ["American IPA"]}
        }

@app.post('/beers/type/', response_model = beersTypeOut)
def predict_beers_type(request_body: beersType):
    data = create_beers_dataset(request_body,ohe,scaler)
    device = get_device()
    prediction = predict(data,model, device,y_encoder)
    print(prediction)
    return prediction

@app.get('/model/architecture/')
def model_architecture():
    values = {"layer_1": str(model.layer_1),
    "layer_2": str(model.layer_2),
    "layer_3": str(model.layer_3),
    "layer_out": str(model.layer_out),
    "softmax": str(model.softmax)}

    return values
    