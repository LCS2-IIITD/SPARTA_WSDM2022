import torch
import torch.nn as nn
from typing import Any, NoReturn



class Classifier(nn.Module):
    """
        A generic classifier module to classify the data (as feaures)
    """
    
    def __init__(self, input_size:int=768, dropout:float=0.10, hidden:list=[768, 512, 256, 128, 64], num_classes=2):
        super(Classifier, self).__init__()
        
        self.input_size = input_size
        self.dropout = dropout
        self.hidden = hidden
        self.num_classes = num_classes
        
        layers = []

        
        for output_size in hidden:
            layers += [
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features=input_size, out_features=output_size),
                nn.LeakyReLU()
            ]
            input_size = output_size
        
        layers.append(nn.Linear(in_features=hidden[-1], out_features=num_classes))
        
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        """
            Forward fuction of generic clasifier, which will take a feacture vector and return a tensor of size num_classes
            Args:
                x: feature vector to classify, x.shape = [batch_size, feature_size]
            Returns:
                outptus of size [batch_size, num_classes]
        """
        outputs = self.classifier(x)
        
        return outputs