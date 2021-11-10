import torch
import torch.nn as nn
from typing import Any, NoReturn
from .Attention import AttentionModel
from .RoBERTa import RepresentationModel
from .SpeakerClassifier import SpeakerClassifierModel
from .MHA import MHAModel
from .Relevance import RelevanceModel
from .GRU import GRUModel
from .Classifier import Classifier



class DACModel(nn.Module):
    
    """
        Complete architecture of the model, all the modules are assmebled in this. 
        
    """
    
    def __init__(self, config:dict)->None:
        
        super(DACModel, self).__init__()
        
        
        self.config = config
        self.model_config = config['model_config'][config['select_model_config']]
        print(self.model_config)
        
        # speaker invariant representation model, attetnion/relevance model and gru block 
        
        # speaker invariant representation model
        self.speaker_invariant = RepresentationModel(
            model_name=config['model_name'], 
            hidden_size=config['hidden_size']
        )

        # speaker-aware  model
        self.speaker_aware = SpeakerClassifierModel(
            config=self.config
        )
        # # load the model from checkpoints and then freeze the weights
        print("Loading Speaker Classifier from Checkpoint ...")
        self.speaker_aware.load_state_dict(torch.load(self.config['speaker_classifier_ckpt_path'])['state_dict'])

        # # freeze the weights of the model if you want
        # if self.speaker_aware is not None:
        #   for param in self.speaker_aware.parameters():
        #     param.requires_grad = False
        # print("Speaker Classifier Loaded and Freezed !")


        
        # # attention block for speaker invaraint pooler
        # if self.model_config['attention_type']=="mha":
        self.si_attention = MHAModel(
            hidden_size=config['hidden_size'], 
            num_heads=config['num_heads'],
            need_weights=config['need_weights'],
            dropout=config['dropout'],
            device=self.config['device'],
        )
        #   # speaker aware attention block
        self.sa_attention = MHAModel(
            hidden_size=config['hidden_size'], 
            num_heads=config['num_heads'],
            need_weights=config['need_weights'],
            dropout=config['dropout'],
            device=self.config['device'],
        )

        # elif self.model_config['attention_type']=="rel":
        # relevance for speaker invariant pooler
        self.si_attention = RelevanceModel(
            hidden_size=config['hidden_size'],
            need_weights=config['need_weights'],
            device=self.config['device'],
        )
        #   # speaker aware relevance block
        self.sa_attention = RelevanceModel(
            hidden_size=config['hidden_size'],
            need_weights=config['need_weights'],
            device=self.config['device'],
        )
        
        
        # gru for speaker invaraint hs
        self.si_gru = GRUModel(
            input_size=config['hidden_size'], 
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            device=self.config['device'],
            
        )

        # speaker aware gru
        self.sa_gru = GRUModel(
            input_size=config['hidden_size'], 
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            device=self.config['device'],
            
        )
        
        
        self.classifier = Classifier(
            input_size=self.model_config['dac_inputs']*config['hidden_size'],
            dropout=self.config['dropout'],
            hidden=self.config['hidden'],
            num_classes=self.config['num_dialogue_acts']
        )

    
    def forward(self,
                batch,

                # previous_sid:str,
                
                si_memory:torch.Tensor=None,
                sa_memory:torch.Tensor=None,
                
                sa_global_context:torch.Tensor=None,
                si_global_context:torch.Tensor=None,
                ):
        """
            forward function of dialogue act classifier
            Args:
                input_ids:input tokens  input_ids.shape = [bs, seq_len]
                attention_mask: masks for tokens, for pad tokens it will be zero for others it will be 1 
                    attention.shape = [bs, seq_len]
                    
                si_memory:speaker invariant fixed window size memory that stores the previous utterances (pooler outputs)
                    si_memory.shape = [window_size, hidden_size]
                sa_memory: speaker aware fixed window size memory that stores the prebious utterances (pooler outputs) 
                    sa_memory.shape = [window_size, hidden_size]
                    
                sa_pooler: speaker aware pooler output of current utterance
                    sa_pooler.shape = [bs, hidden_size]
                sa_hidden_states: speaker aware hidden states of current utterance
                    sa_hidden_states.shape = [bs, seq_len, hidden_size]
                    
                mem_attention_mask:attention mask for memory if we have history less than the window size some of the them 
                will be masked.
                    mem_attenion_mask.shape = [windows_size]
                sa_global_context: global context (initial hidden state) for speaker aware GRU
                    sa_global_context.shape = [bs, hidden_size]
                si_global_context: global context (initial hidden state) for speaker invariant GRU
                    sa_global_context.shape = [bs, hidden_size]
            Returns:
                
        """
      
        # adjusted during experiment
        previous_sid="none"
        
        batch_size = batch['input_ids'].shape[0]        
        

        """Speaker Invariant Model"""

        # get the speaker aware representations 
        si_hidden_states, si_pooler = self.speaker_invariant(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )

 
        # feature and attention vector for si attention block
        si_attn_x = torch.empty((0, self.config['hidden_size']), device=self.config['device'])
        si_attn = torch.empty((0, self.config['window_size']), device=self.config['device'])

    
        for i, x in enumerate(zip(si_pooler.detach(), batch['session_id'])):

            si_utterance, current_sid = x[0].unsqueeze(0), x[1]

            xi_attn, attn_i = self.si_attention(
                memory=si_memory,
                current_utterance=si_utterance,
            )
            si_attn_x = torch.vstack((si_attn_x, xi_attn))
            si_attn = torch.vstack((si_attn, attn_i.squeeze(0)))

            # update the memory  
            si_memory = torch.vstack((si_utterance, si_memory[:-1]))

        si_gru_x, si_global_context = self.si_gru(
              embedding=si_hidden_states,
              global_context=si_global_context,
              attention_mask=batch["attention_mask"],
              session_ids=batch['session_id'],
              previous_sid=previous_sid
          )

        # """Speaker Aware Model"""


        # # feature and attention vector for sa attention block
        sa_attn_x = torch.empty((0, self.config['hidden_size']), device=self.config['device'])
        sa_attn = torch.empty((0, self.config['window_size']), device=self.config['device'])


        # # for speaker aware we are going to take all speaker aware components including local context, global context and speaker aware invariant representation
          
        # get speaker invariant representation
        sa_hidden_states, sa_pooler, _ = self.speaker_aware(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )

        for i, x in enumerate(zip(sa_pooler.detach(), batch['session_id'])):
            
            sa_utterance, current_sid = x[0].unsqueeze(0), x[1]
                            
            xa_attn, attn_a = self.sa_attention(
                memory=sa_memory,
                current_utterance=sa_utterance,
            )

            sa_attn_x = torch.vstack((sa_attn_x, xa_attn))
            sa_attn = torch.vstack((sa_attn, attn_a.squeeze(0)))

            # update the speaker aware memory 
            sa_memory = torch.vstack((sa_utterance.to(self.config['device']), sa_memory[:-1].to(self.config['device'])))
            
        sa_gru_x, sa_global_context = self.sa_gru(
            embedding=sa_hidden_states,
            global_context=sa_global_context,
            attention_mask=batch["attention_mask"],
            session_ids=batch['session_id'],
            previous_sid=previous_sid
        )
        
        
        
        # concat all the features vector
        features = torch.empty((batch_size, 0), device=self.config['device'])
        
      
        
        for i, x in enumerate([si_pooler, si_attn_x, si_gru_x, sa_pooler, sa_attn_x, sa_gru_x]):
            if x.shape[0]==batch_size:
                features = torch.hstack((features, x))
        

        logits = self.classifier(features)

        
        
        # return updated memories, attentions and logits

        return {
            # memory
            "si_memory":si_memory,
            "sa_memory":sa_memory,
            
            "si_global_context":si_global_context,
            "sa_global_context":sa_global_context,

            "si_attn":si_attn,
            "sa_attn":sa_attn,

            # logits to compute the loss
            "logits":logits,
            
            # "previous_sid":current_sid
        }
        