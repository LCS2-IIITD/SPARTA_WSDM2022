import torch
import torch.nn as nn
from typing import Any, NoReturn


class RelevanceModel(nn.Module):
    """
        Relvance model is somehwhat similar to attention but it a scaled by a distance measure
        Params:
            hidden_size: size of the key, query, values hidden vectors
            need_weights: should we apply linear projection or not
    """
    
    def __init__(self, hidden_size:int=768, need_weights:bool=True, device=torch.device("cpu"))->None:
        
        super(RelevanceModel, self).__init__()
        
        self.device = device
        
        self.hidden_size = hidden_size
        self.need_weights = need_weights
        
        
        
        # linear layers to linearly project key, query and values and weighted sum of values where attention as weight 
        self.k = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.q = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.v = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.o = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        
    def distance_fn(self, d:int, T:float=0.20)->torch.Tensor:
        """
            Function to compute the distance weights
            Args:
                d: distance (a single element tensor)from current utterance to that utterance
                T: a hyperparameter, to cool down the distance function, T basically controls the distance weight 
                as well similarity weights.
                if T is low max value of distance will be high as well as max-min will be low. I have tested the T 
                for different range. 
                T seems good between range 0.10 - 0.30 anything below 0.50 is tolerable
            
            Returns: distance weight as tensor
            
        """
        return 1/(torch.exp(torch.tensor([d]))**T) # different functions have been tested but simple inverse was used finally
        
    
    def forward(self, memory:torch.Tensor, current_utterance:torch.Tensor, attention_mask:torch.Tensor=None)->tuple:
        """
            forward function of Relvance which takes keys queries and values and returns the weighted sum 
            Arguments:
                memory: memory in which previous utterances representations are stored, memory.shape = [window_size, embedding_size]
                current_utterance: current utterance repsentation , current_utterance.shape = [1, embedding_size]
                attention_mask: attention over memory, initially it all be set to 0 but as we will fill the memory it will 
                be unmasked. 
            
            Returns: attention maps and contextualized embedding as tuple
        """
        window_size = memory.shape[0]
        
        # treat memory and current_utterance as key, value and query
        key = memory.to(self.device) # [1, memory_size, hidden_size]
        query = current_utterance.to(self.device) # [1, 1, hidden_size]
        value = memory.to(self.device) # [1, memory_size, dden_size]
        
        if self.need_weights:
            key = self.k(key)
            value = self.v(value)
            query = self.q(query)
      
        
        
        # computing similarity function between memory 
        similarity = torch.cosine_similarity(key, query)
        # similarity.shape = [window_size]
        
        # get the distance weights with the help of distance function defined 
        distance = torch.linspace(start=1, end=window_size, steps=window_size, dtype=torch.float).apply_(self.distance_fn).to(self.device)
        # distance.shape = [window_size]
        
        # compute the energy with the help of similarity and distance which is product of these two
        energy = similarity*distance
        #energy.shape [window_size]
        
        # mask the not use full context
        if attention_mask is not None:
            energy = energy.masked_fill(attention_mask == 0, -1e10)
            
        
        # compute the relevance score by applying softmax over energy
        relevance = energy.softmax(dim=-1)
        # relevance.shape [window_size]
        
        # get the weighted average of values where relevance is weight
        x = torch.matmul(relevance.unsqueeze(0), value)
        
        return x, relevance
   
 