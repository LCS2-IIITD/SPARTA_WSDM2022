import torch
import torch.nn as nn
from typing import Any, NoReturn


class MHAModel(nn.Module):
    """
            This is pytorch's implementation of Multihead Attention 
            Parameters:
                embedding_size: What is size of input embedding
                num_heads: Number of heads we want, embedding size should be divisible by num_heads
                need_weights: Whether we want to project the key, query and value vectors or not
        """
    
    def __init__(self, hidden_size:int=768, num_heads:int=12, need_weights:bool=True, dropout:float=0.10, device=torch.device("cpu")):
        """
            This is pytorch's MHA model
            Parameters:
                embedding_size: What is size of input embedding
                num_heads: Number of heads we want, embedding size should be divisible by num_heads
                need_weights: Whether we want to project the key, query and value vectors or not
        """
        
        super(MHAModel, self).__init__()
    
        self.device = device
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.need_weights = need_weights
        
        # multihead attention model
        self.mha = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=dropout,
        )
        
    def forward(self, memory:torch.Tensor, current_utterance:torch.Tensor, attention_mask:torch.Tensor=None)->tuple:
        """
            forward function of MHA which takes following argumennts
            Arguments:
                memory: memory in which previous utterances representations are stored, memory.shape = [window_size, embedding_size]
                current_utterance: current utterance repsentation , current_utterance.shape = [1, embedding_size]
                attention_mask: attention vector for keys to avoid paying attention to pad tokens
            
            Returns: attention maps and contextualized embedding as tuple
            
        Note: Since this will be applied to to memory and single utterance at a time 
        
        Convert the memory and current utterance as key query vector
        key: keys to compute attention weights, key.shape = [bs, k_len, embedding_size]
        query: query for which we need context vector, query.shape = [bs, q_len, embedding_size]
        value: values to get contextualized query, value.shape = [bs, v_len, embedding_size]
        
        """
        
        # treat memory and current_utterance as key, value and query
        key = memory.unsqueeze(0).to(self.device) # [1, memory_size, hidden_size]
        query = current_utterance.unsqueeze(0).to(self.device) # [1, 1, hidden_size]
        value = memory.unsqueeze(0).to(self.device) # [1, memory_size, dden_size]
        
        x, attention  = self.mha(
            key=key,
            query=query,
            value=value,
            need_weights=self.need_weights
        )
        
        return x.squeeze(0), attention
        
        
        