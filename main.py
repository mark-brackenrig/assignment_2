from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd



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
    
    
    