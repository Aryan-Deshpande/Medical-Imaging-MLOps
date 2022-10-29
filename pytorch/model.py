from model_class import MLE
import numpy as np
import pandas as pd
import torch

def load_model():

    state_dict = torch.load('model2.pt', map_location="cpu")
    model = MLE()

    model.load_state_dict(state_dict=state_dict)
    model.eval()

    return model

def inference(device, time, model): # here time will have ot converted to a model interpretable format
    
    data = torch.tensor([device, time], dtype=torch.float32)
    data = data.unsqueeze(0)

    with torch.no_grad():
        out = model(data)

    return out



