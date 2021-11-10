import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, NoReturn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report  
from .dataset.dataset import DADataset
from datasets import load_dataset
from torch.utils.data import DataLoader




class LightningModel(pl.LightningModule):
    
    def __init__(self, model, tokenizer, config):
        super(LightningModel, self).__init__()
        
        self.config = config
        self.model_config = config['model_config'][config['select_model_config']]

        
        
        self.model = model
        self.tokenizer = tokenizer

        
        self.si_memory = torch.randn((self.config['window_size'], self.config['hidden_size']), device=self.config['device'])
        self.sa_memory = torch.randn((self.config['window_size'], self.config['hidden_size']), device=self.config['device'])
        self.si_global_context = None
        self.sa_global_context = None
        
        print(f'Total trainable parameters = {self.count_parameters()}')

        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, batch):
    
        outputs = self.model(
            batch=batch,
            si_memory=self.si_memory,
            sa_memory=self.sa_memory,
            sa_global_context=self.sa_global_context,
            si_global_context=self.si_global_context,
        )
        
        
        # # update the memory and global context
        # # self.si_memory = outputs['si_memory']
        self.si_global_context = outputs['si_global_context']
        self.sa_global_context = outputs['sa_global_context']
        
        self.sa_memory = outputs['sa_memory']
        self.si_memory = outputs['si_memory']
        

        return outputs['logits']
    
    
    def configure_optimizers(self):

        return optim.Adam(params=self.parameters(), lr=self.config['lr'])
    
    
    def train_dataloader(self):
        train_data = load_dataset("csv", data_files=self.config['data_dir']+self.config['data_files']['train'])
        
        train_dataset = DADataset(tokenizer=self.tokenizer, data=train_data['train'], fields=self.config['fields'], max_len=self.config['max_len'])
        
        ## create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        
        
        return train_loader 
    
    
    def training_step(self, batch, batch_idx):
        targets = batch['label']
        logits = self(batch)
        
        loss = F.cross_entropy(logits, targets)
        
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        wandb.log({"loss":loss, "accuracy":acc, "f1_score":f1})
        return {"loss":loss, "accuracy":acc, "f1_score":f1}
        
    def val_dataloader(self):
        valid_data = load_dataset("csv", data_files=self.config['data_dir']+self.config['data_files']['valid'])
        valid_dataset = DADataset(tokenizer=self.tokenizer, data=valid_data['train'], fields=self.config['fields'], max_len=self.config['max_len'])
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        return valid_loader

    def validation_step(self, batch, batch_idx):

        targets = batch['label']

        logits = self(batch)

        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        precision = precision_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        recall = recall_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        return {"val_loss":loss, "val_accuracy":torch.tensor([acc]), "val_f1":torch.tensor([f1]), "val_precision":torch.tensor([precision]), "val_recall":torch.tensor([recall])}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_precision = torch.stack([x['val_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['val_recall'] for x in outputs]).mean()
        wandb.log({"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1, "val_precision":avg_precision, "val_recall":avg_recall})
        return {"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1, "val_precision":avg_precision, "val_recall":avg_recall}

    def test_dataloader(self):
        test_data = load_dataset("csv", data_files=self.config['data_dir']+self.config['data_files']['test'])
        test_dataset = DADataset(tokenizer=self.tokenizer, data=test_data['train'], fields=self.config['fields'], max_len=self.config['max_len'])
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        return test_loader

    def test_step(self, batch, batch_idx):

        targets = batch['label']

        logits = self(batch)

        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        precision = precision_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        recall = recall_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        return {"test_loss":loss, "test_precision":torch.tensor([precision]), "test_recall":torch.tensor([recall]), "test_accuracy":torch.tensor([acc]), "test_f1":torch.tensor([f1])}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        avg_precision = torch.stack([x['test_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['test_recall'] for x in outputs]).mean()
        return {"test_loss":avg_loss, "test_precision":avg_precision, "test_recall":avg_recall, "test_acc":avg_acc, "test_f1":avg_f1}


    
    