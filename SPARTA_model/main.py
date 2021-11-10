from config import config
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer
import pytorch_lightning as pl
from Trainer import LightningModel
from .models.DAC import DACModel

if __name__ == '__main__':
    
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config['model_name'])
    logger = WandbLogger(
        name="analyze"+"mha-final",
        save_dir=config["save_dir"],
        project=config["project"],
        log_model=True,
    )
    early_stopping = EarlyStopping(
        monitor=config["monitor"],
        min_delta=config["min_delta"],
        patience=config['patience'],
    )
    checkpoints = ModelCheckpoint(
        filepath=config["filepath"],
        monitor=config["monitor"],
        save_top_k=1
    )
    ## Trainer
    trainer = pl.Trainer(
        logger=logger,
        gpus=[0],
        checkpoint_callback=checkpoints,
        callbacks=[early_stopping],
        default_root_dir="./",
        max_epochs=config["epochs"],
        precision=config["precision"],
        enable_pl_optimizer=False,
        automatic_optimization=True,
    )
    
    model = DACModel(config=config)
    lm = LightningModel(
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    trainer.fit(lm)
    trainer.test(lm)
    
    
