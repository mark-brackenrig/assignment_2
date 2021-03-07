import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from src.data import make_dataset

def create_dataset(request_body, ohe, scaler):
    X = pd.DataFrame(request_body)
    brewery_names = ohe.transform(X['brewery_names'])
    X.drop('brewery_names',inplace = True)
    X = scaler.fit_transform(X)
    X = np.concatenate([X,brewery_names],axis = 1)

def predict(test_data, model,y_encoder, criterion, batch_size, device, generate_batch=None):
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
    Dictionary
        Prediction: index of output
        
    """    
    
    # Set model to evaluation mode
    model.eval()
    test_loss = 0
    test_acc = 0
    
    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size, collate_fn=generate_batch)
    results = []
    # Iterate through data by batch of observations
    for feature in data:
        
        feature = feature.to(device)
        
        # Set no update to gradients
        with torch.no_grad():
            # Make predictions
            output = model(feature)
        
        class_name = make_dataset.inverse_transformer(y,y_encoder)

        results.append({'index': output.argmax(1)[0], 'class': class_name})
    return results

