from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import torch
import pickle as pkl

ohe = pkl.load(open('models/ohe.pkl','rb'))
y_encoder = pkl.load(open('models/y_encoder.pkl','rb'))
model = torch.load('models/beer_style_prediction.pt')

app = FastAPI()

@app.get('/')
def display_project_description():
    """
        Displaying a brief description of the project objectives, list of endpoints, expected input parameters  and output format of the model, link to the Github repo related to this project
    """
        
    return {'Hello World Placeholder'}

@app.get('/health/')
def hello_world():
    print("Hello World")
    return {"Hello World! App running."}
    
@app.post('/beer/type/')
def predict_beer_type():
    print("Hello World")
    return {"Hello World! App running."}

@app.post('/beers/type/')
def predict_beers_type():
    print("Hello World")
    return {"Hello World! App running."}

@app.get('/model/architecture/')
def predict_beers_type():
    print("Hello World")
    return {"Hello World! App running."}
    