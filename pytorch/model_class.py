import numpy as np
import pandas as pd
import torch 
import torch.nn as nn

class MLE(nn.Module):
  def __init__(self):
    super(MLE,self).__init__()

    self.model = nn.Sequential(
        nn.Linear(2,1)
    )
  
  def forward(self,x):
    x.unsqueeze(0)
    out = self.model(x)
    return out