import torch
import torch.nn as nn
from typing import Any, NoReturn
from .RoBERTa import RepresentationModel
from .Classifier import Classifier

class SpeakerClassifierModel(nn.Module):
    """Speaker Classifier model, isolated from complete architecture, easy ablation study"""
    
    def __init__(self, config:dict):
        super(SpeakerClassifierModel, self).__init__()
        
        self.config = config
        
        self.base = RepresentationModel(
            model_name=config['model_name'],
            hidden_size=config['hidden_size']
        )
        
        self.classifier = Classifier(
            input_size=config['hidden_size'],
            dropout=config['dropout'],
            num_classes=config['num_speakers']
        )
        
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor=None)->tuple:
        """
            forward function of Speaker Classifier model 
            Args:
                input_ids:input tokens  input_ids.shape = [batch, seq_len]
                attention_mask: masks for tokens, for pad tokens it will be zero for others it will be 1
            Returns: a tuple containing hidden states, pooler output and logits
        """
        
        # get the repsentation
        hidden_states, pooler = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # classify the speakers
        logits = self.classifier(pooler)
        
        return hidden_states, pooler, logits
        
        