import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from src.data import make_dataset
import pandas as pd
from src.models.pytorch import get_device, PytorchDataset

def create_dataset(request_body, ohe, scaler):
    """
    Converts the request body of the API to a numpy array that can be read by the model.
    Parameters
    ----------
    request_body: dict
    the request body from the API 
    """
    X = pd.DataFrame(request_body)
    X.columns = ['key','value']
    X.set_index('key', inplace = True)
    X = X.transpose().reset_index(drop=True)
    val = np.array(X['brewery_names']).reshape(-1,1)
    brewery_names = ohe.transform(val)
    X.drop(columns = 'brewery_names',inplace = True, axis = 1)
    X = scaler.transform(X)
    X = np.concatenate([X,brewery_names],axis = 1)
    return X

def predict(test_data, model, device,y_encoder, generate_batch=None):
    """Calculate performance of a Pytorch multi-class classification model

    Parameters
    ----------
    test_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """    
    # Set model to evaluation mode
    model.eval()
    test_loss = 0
    test_acc = 0
    
    X = torch.Tensor(test_data)
    # Create data loader
    data = DataLoader(X, batch_size=1, collate_fn=generate_batch)
    results = []
    # Iterate through data by batch of observations
    for feature in data:
        
        feature = feature.to(device)
        
        # Set no update to gradients
        with torch.no_grad():
            # Make predictions
            output = model(feature)
        print(output)
        prediction = output.argmax(1)[0].item()
        class_name = make_dataset.inverse_transformer(np.array(prediction).reshape(-1,1),y_encoder) 
        print(class_name.flatten())
        results.append({'class_name':class_name.flatten().tolist()})
    return results

