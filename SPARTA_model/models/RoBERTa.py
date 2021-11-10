import torch
import torch.nn as nn
from typing import Any, NoReturn
from transformers import AutoConfig, AutoModel
class RepresentationModel(nn.Module):
    """
        Represention Model to get the contextualized vectors (hidden states) and sentence representation (pooler output)
        
        Parameters:
            model_name: name of pretrained model as feature extractor or represenation model
        
        
    """
    
    def __init__(self, model_name:str, hidden_size:int=768)->None:
        super(RepresentationModel, self).__init__()
        
        # feature extractor with default config 
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, hidden_size=hidden_size)
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name, config=self.config)
        
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor=None)->tuple:
        """
            forward function of representation model which takes input_ids and attention_mask(optional)and gives contextualized 
            embeddings.
            
            Arguments:
                input_ids:input ids tensor
                attention_mask: attention maks, 0 for pad tokens 1 for sequence tokens
                
            Returns:
                hidden states and pooler output in that order
        """
        
        # feed the input_ids and attention to the representation model 
        hidden_states, pooler = self.base(input_ids=input_ids, attention_mask=attention_mask)
        return hidden_states, pooler
    
        