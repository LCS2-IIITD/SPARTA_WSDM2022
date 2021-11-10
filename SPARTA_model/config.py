import torch
config = {
    # data
    "data_dir":"./drive/MyDrive/a2g-merged/",
    "data_files":{
        "train":"a2g_train.csv",
        "valid":"a2g_valid.csv",
        "test":"a2g_test.csv",
    },
    # field to read from the dataset, its is limited to our dataset only
    "fields":{
        "text":"Utterance",
        "act":"Dialog_Act",
        "label":"Dialog_Act_Label",
        "id":"ID",
        "speaker":"Type"
    },
    
    "max_len":512,
    "batch_size":8,
    "num_workers":6,
    
    "previous_sid":"start",
    
    # models
    "model_name":"roberta-base",
    "hidden_size":768,
    "num_layers":1,
    "num_heads":12,
    "dropout":0.15,
    "need_weights":True,
    "start_sid":"start",
    "window_size":6,

    "speaker_classifier_ckpt_path":"./drive/MyDrive/speaker-classifier/classifier.ckpt",
    
    "num_speakers":2,
    
    "hidden":[1024, 768, 512, 256, 128, 64],
    
    "num_dialogue_acts":10,
    
    "model_config":None, #model_config,
    "select_model_config":-1, # it will be from [0, 1, 2, 3, 4, 5, 6, 7]
   
    
    
    # training config
    # training
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_dir":"./",
    "project":"dialogue-act-classification",
    
    "lr":1e-5,
    "monitor":"val_f1",
    "min_delta":0.001,
    "patience":5,
    "filepath":"./models/{epoch}-{val_accuracy:4f}",
    "precision":32,
    "epochs":50,
    "average":"macro"
    
}