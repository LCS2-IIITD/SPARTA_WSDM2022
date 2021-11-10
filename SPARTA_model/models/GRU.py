import torch
import torch.nn as nn
from typing import Any, NoReturn




class GRUModel(nn.Module):
    
    """ 
        GRU model that takes the global context along with current utterance representation and gives a context vector 
        
        Parameters:
            input_size: size of the input to GRU, which will be contextualized embedding means hidden size
            hidden_size: size of the GRU's hidden vectors
            num_layer: how many layers should be there in GRU
            bidirectional: whether bidirectional or not if True both forwar and backward hidden states will be concatented
                  
    """
    
    def __init__(self, input_size:int=768, hidden_size:int=768, num_layers:int=1, bidirectional:bool=False, max_len:int=512, device=torch.device("cpu"))->None:
        super(GRUModel, self).__init__()
        
        self.max_len = max_len

        self.device = device
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )        
        
    
    def forward(self, embedding:torch.Tensor, attention_mask:torch.Tensor, session_ids:list, previous_sid:str, global_context:torch.Tensor=None)->tuple:
        """
            forward function of GRUModel which will take contextualized embeddings and global context as input
            and will produce next hidden states.
            
            Arguements:
                embedding: contextextualized embeddings (hidden states) coming from representation model [bs, seq_len, hidden_size]
                attention_mask: To ignore the pad tokens 
                global_context: last hidden state from previous utterance [bs, num_direction, hidden_size]
            
            Returns:
                gloabl_context: last hidden state of current utterance
                
        """
        bs, seq_len, hidden_size = embedding.shape[0], embedding.shape[1], embedding.shape[2]
        
        outputs = torch.empty((0, hidden_size), device=self.device)

        if global_context is None:
          global_context = torch.randn((1, 1, hidden_size), device=self.device)


        for i, x in enumerate(zip(embedding, attention_mask, session_ids)):
            
           
            
            input_, current_sid = x[0],  x[2]
            
            input_len = self.max_len - torch.unique(x[1], return_counts=True)[1][0].item()
            if input_len < 1:
                input_len = 2
            
            # padded tokenz remove mannual with fixed len
            _, global_context = self.gru(input=input_[:input_len].unsqueeze(0), hx=global_context.detach())

            
            outputs = torch.vstack((outputs, global_context.squeeze()))
            
      
        return (outputs, global_context.detach())
        
        
    
    
    