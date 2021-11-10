import torch
from typing import Any, NoReturn
from torch.utils.data import Dataset
class DADataset(Dataset):
    """
        Custom Dataset class to create the DA dataset
        
        Arguments:
            tokenizer: tokenizer to tokenize the utterance
            data: data either a df or hugginface dataset 
            train: whether its training set or not if True we will build the label dict else not 
            text_field: text field in the dataframe 
            label_field: label field in the dataframe
            max_len: maximum sequence  length, truncation and padding will be applied
    """
    
    __label_dict = dict()
    __speaker_dict = {
        "T":0,
        "P":1,
    }
    
    def __init__(self, 
                 tokenizer:Any, 
                 data:Any, 
                 fields:dict, 
                 max_len:int=512)->None:
        
        self.text = data[fields['text']]
        self.act = data[fields['act']]
        self.label = data[fields['label']]
        self.speaker = data[fields['speaker']]
        self.ids = data[fields['id']]
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # build label and speaker dict dictionary 
        for i in range(len(self.act)):
            if self.act[i] not in DADataset.__label_dict.keys():
                DADataset.__label_dict[self.act[i]]=self.label[i]
            
    
    def __len__(self)->int:
        return len(self.text)
    
    def label_dict(self)->dict:
        """
            label_dict method get the label vocab
            Returns:
                label dict, a dictionary containing class name as keys and indices as values
        """
        
        return DADataset.__label_dict
    
    
    def speaker_dict(self)->dict:
        """
            label_dict method get the label vocab
            Returns:
                label dict, a dictionary containing class name as keys and indices as values
        """
        
        return DADataset.__speaker_dict
    
    
    
    def __getitem__(self, index:int)->dict:
        """
            __getitem__() method to get a sample from dataset
            
            Parameters:
                index: index of datasets to fetch the utterance
                
            Returns:
                item as dictionary 
        """
        
        text = self.text[index]

        act = self.act[index]
        label = self.label[index]
        speaker = DADataset.__speaker_dict[self.speaker[index]]
        session_id = self.ids[index].split("_")[0]
        
        
        # encode the text
        input_encoding = self.tokenizer.encode_plus(
            text=text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            return_length=True
            
        )
        
        return {
            "text":text,
            "input_ids":input_encoding['input_ids'].squeeze(),
            "attention_mask":input_encoding['attention_mask'].squeeze(),
            "act":act,
            "label":torch.tensor([label], dtype=torch.long).squeeze(),
            "speaker":torch.tensor([speaker], dtype=torch.long).squeeze(),
            "session_id":session_id
            
        }