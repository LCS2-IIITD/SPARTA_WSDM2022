import torch
import torch.nn as nn
from typing import Any, NoReturn


class AttentionModel(nn.Module):
    """
        Attention block to pay attention to dialouge history, this is multihead attention mechanism if num_heads = 1, it will be 
        context-context dot product attention.
        
        Parameters:
            hidden_size: What is size of input vector
            num_heads: Number of heads we want, embedding size should be divisible by num_heads
            need_weights: Whether we want to project the key, query and value vectors or not
            
    """
    def __init__(self, hidden_size:int=768, num_heads:int=1, need_weights:bool=True)->None:
        super(AttentionModel, self).__init__()
        
        assert hidden_size%num_heads==0, "hidden size should be divisible by nums of heads"
        
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size//num_heads
        self.need_weights = need_weights
        
        # projection weights
        self.k = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.q = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.v = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.o = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        
        #  no scalong as of now
        
        
    
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
        key = memory.unsqueeze(0) # [1, memory_size, hidden_size]
        query = current_utterance.unsqueeze(0) # [1, 1, hidden_size]
        value = memory.unsqueeze(0) # [1, memory_size, dden_size]
        
        batch_size = key.shape[0]
        
        
        
        if self.need_weights:
            key = self.k(key)
            query = self.q(query)
            value = self.v(value)
      
        # divide the key, qeury and value into heads
        # key.shape, query.shape, value.shape = [bs, _len, embedding_size]
        K = key.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        Q = query.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = value.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        # K.shape, Q.shape, V.shape = [bs, num_heads, _len, head_size]
        
        
        # compute the attention weights
        weights = torch.matmul(Q, K.permute(0, 1, 3, 2))
        # weights.shape = [bs, num_heads, q_len, k_len]
        
        
        ##[4, 12, 1, 512] to [4, 1, 1, 512]
        if attention_mask is not None:
            # fill the masked position with small values (tends to zero) in order to pay zero attention
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            weights = weights.masked_fill(attention_mask == 0, -1e10)
        
        # apply the softmax over attention weights to normalize it
        attention = torch.softmax(weights, dim=-1)
        # attention.shape = [bs, num_heads, q_len, k_len]
        
        
        # weighted sum of values where attention should be used as weights, k and query should be of same length
        x = torch.matmul(attention, V)
        # x.shape = [bs, num_heads, q_len, head_size]
        
        
        x = x.permute(0, 2, 1, 3).contiguous()
        # x.shape = [bs, q_len,  num_heads, head_size]
        
        
        x = x.view(batch_size, -1, self.hidden_size)
        # x.shape = [bs, q_len, embedding_size] (original size of query)
        
        
        # fed it to linear layer if need_weights is True
        if self.need_weights:
            x =  self.o(x)
        
        ##print(f'x.shape after linear = {x.shape}')
        
        return x.squeeze(0), attention
        