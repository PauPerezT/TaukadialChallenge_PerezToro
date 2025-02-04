"""
Taukadial Challenge

->model.py

Created on Thu Jan 18 18:08:00 2024 for the Taukadial Challenge

@author: This code was created by Paula A. Pérez-Toro
@email:paula.andrea.perez@fau.de
"""

import gc

import torch
import json
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import nltk
import spacy
import string
import evaluate  # Bleu
import pandas as pd
import numpy as np
import transformers
import matplotlib.pyplot as plt

import warnings
from transformers import WhisperModel
import pytorch_lightning as pl
from transformers import AdamW
from torchmetrics.text.bert import BERTScore
from torchmetrics.text import BLEUScore, ROUGEScore
from data import get_dataset

import torch
import wandb

from torch.utils.data import DataLoader
#LoginWandB


from pytorch_lightning import seed_everything
import os
from torchmetrics import Accuracy, Recall, Specificity, AUROC, ConfusionMatrix, F1Score
from torchmetrics.classification import BinaryRecall, BinarySpecificity
from sklearn.metrics import classification_report,precision_recall_fscore_support

def metrics_sk(preds, scores, labels):
    class_names= ['NC', 'MCI']
    dict = {}
    classes = []
    #print(preds,  labels)
    print('precision_recall_fscore_support',precision_recall_fscore_support(labels, preds, average=None, labels=np.hstack([0, 1])))
    for i, c in enumerate(precision_recall_fscore_support(labels, preds, average=None, labels=np.hstack([0, 1]))[1]):
        print(class_names[i], c)
        classes.append(c)
        dict[class_names[i]] = c

    uar = np.average(classes)
    dict['UAR'] = uar

    #fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    #auc = metrics.auc(fpr, tpr)

    return dict
seed_everything(42, workers=True)

warnings.filterwarnings("ignore")


#----------------------------

class Whp_Ti(pl.LightningModule):
    def __init__(self, num_labels=2, class_weights =None):
        super(Whp_Ti, self).__init__()

        #self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.pred_val = []  #
        self.real_val = []
        self.scores_val = []

        self.pred = []
        self.scores = []
        self.real = []

        #self.emb = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.emb_en = WhisperModel.from_pretrained("openai/whisper-large")#,output_hidden_states=True) #jonatasgrosman/whisper-large-zh-cv11 #
        self.emb_zh = WhisperModel.from_pretrained("jonatasgrosman/whisper-large-zh-cv11")#,output_hidden_states=True) #jonatasgrosman/whisper-large-zh-cv11 #
        self.class_weights = class_weights.to(self.emb_en.device)
        self.drop = nn.Dropout()
        self.linear1 = nn.Linear(1280, 128)
        self.linear_tim =nn.Linear(17,128)
        self.linear2 = nn.Linear(128,num_labels)

        # Frozen Model
        for name, param in self.emb_en.named_parameters():
            param.requires_grad = False
        for name, param in self.emb_zh.named_parameters():
            param.requires_grad = False


        if num_labels>2:
            class_type = "multiclass"
            self.class_names = ['Control', 'Mild', 'Moderate', 'Severe']
        else:
            class_type = "binary"
            self.class_names = ['NC', 'MCI']
            self.metrics = nn.ModuleDict({
                'F1Score': F1Score(task=class_type, average='macro', num_classes=num_labels),
                'Specificity': BinarySpecificity(),
                'Sensitivity': BinaryRecall(),
                'Acc': Accuracy(task=class_type, num_classes=num_labels, average='macro'),
                # 'AUC': AUROC(task=class_type, num_classes=num_labels)
                # 'CM':  ConfusionMatrix(task = class_type, num_classes = num_labels)
            })

            # self.id2label = {id: label for id, label in enumerate(self.class_names)}
            # self.label2id = {label: id for id, label in self.id2label.items()}
            # self.AUC = AUROC(task=class_type, num_classes=num_labels)

    def forward(self, values):
        decoder_input_ids = torch.tensor([[1, 1]]) * self.emb_en.config.decoder_start_token_id
        decoder_input_ids_zh = torch.tensor([[1, 1]]) * self.emb_zh.config.decoder_start_token_id
        #print(values['input_ids'].shape, decoder_input_ids.shape, decoder_input_ids_zh.shape)
        ids_zh = []
        ids_en = []
        for  i,lg in enumerate(values['language']):
            if lg =='zh':
                ids_zh.append(i)
            else: 
                ids_en.append(i)
        
        if len(ids_zh)!=0 and len(ids_en)!=0:
            outputs_zh = self.emb_zh(values['input_ids'][np.hstack(ids_zh)],decoder_input_ids=decoder_input_ids_zh.to(self.emb_zh.device)).last_hidden_state #.encoder_hidden_states

            outputs_en = self.emb_en(values['input_ids'][np.hstack(ids_en)],decoder_input_ids=decoder_input_ids.to(self.emb_en.device)).last_hidden_state #.encoder_hidden_states
            
            outputs = torch.stack((outputs_zh,outputs_en))
        elif len(ids_zh)!=0:
            outputs = self.emb_zh(values['input_ids'][np.hstack(ids_zh)],decoder_input_ids=decoder_input_ids_zh.to(self.emb_zh.device)).last_hidden_state #.encoder_hidden_states

        elif len(ids_en)!=0:
            outputs = self.emb_en(values['input_ids'][np.hstack(ids_en)],decoder_input_ids=decoder_input_ids.to(self.emb_en.device)).last_hidden_state#.encoder_hidden_states

        #print(outputs.shape)
        outputs = torch.mean(outputs, dim=1)
        outputs = self.drop(self.linear1(outputs))
        out_tim = self.linear_tim(values['Timing'])
        preds = self.linear2(outputs+out_tim)

        return {'preds': preds , 'labels':  values["labels"]}

    def common_step(self, batch, batch_idx, num_labels=75):
        outputs = self(batch)
        loss = self.compute_loss(batch, outputs)

        # metrics = metrics_sk(preds, labels)

        # print(outputs['preds'])
        sft = nn.functional.softmax(outputs['preds'], dim=1)

        preds = torch.max(sft, dim=1)[1]

        scores = sft[:, 1]
        print(preds, outputs['labels'])
        metrics = metrics_sk(preds.cpu().numpy(), scores.detach().cpu().numpy(),
                                outputs['labels'].cpu().numpy())

        return loss, metrics, preds, scores, outputs['labels']

    def compute_loss(self, batch, outputs):
        #cse = nn.CrossEntropyLoss(weight=self.weights.to(self.bert.device)).to(self.bert.device)
        cse = nn.CrossEntropyLoss()
        loss = cse(outputs['preds'], outputs['labels'])

        return loss

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics, preds, scores, real = self.common_step(batch, batch_idx)
        # scores = self.eval_common_step(batch)
        # self.log_dict({'val_loss': loss, **
        # {k+'_val': v for k, v in scores.items()}}, on_step=False, on_epoch=True, sync_dist=True)
        self.pred_val.append(preds.cpu().numpy())
        self.real_val.append(real.cpu().numpy())
        self.scores_val.append(scores.cpu().numpy())
        #self.log_dict({'val_loss': loss, **
        #{k + '_val': v for k, v in metrics.items()}}, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss, preds.cpu().numpy(), scores.detach().cpu().numpy(), real.cpu().numpy()

    def on_validation_epoch_end(self):
        metrics = metrics_sk(np.hstack(self.pred_val), np.hstack(self.scores_val),
                                np.hstack(self.real_val))
        self.log_dict({**{k + '_val': v for k, v in metrics.items()}}, on_step=False, on_epoch=True, sync_dist=True)

        self.pred_val = []  #
        self.real_val = []
        self.scores_val = []





    def test_step(self, batch, batch_idx):
        loss, metrics, preds, scores, real = self.common_step(batch, batch_idx)

        self.pred.append(preds.cpu().numpy())
        self.scores.append(scores.detach().cpu().numpy())
        self.real.append(real.cpu().numpy())

        #self.log_dict({'test_loss': loss, **
        #{k + '_test': v for k, v in metrics.items()}}, on_step=False, on_epoch=True, sync_dist=True)

        # , 'CM_test': wandb.plot.confusion_matrix(probs=None, y_true=self.id2label(batch['labels']), preds=self.id2label(preds),class_names=self.class_names)},
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        #return loss, preds.cpu().numpy(), scores.detach().cpu().numpy(), real.cpu().numpy(), batch['age'].cpu().numpy(), batch['gender'].cpu().numpy(), batch['gender'].cpu().numpy()
        return loss

    def on_test_epoch_end(self):
        metrics = metrics_sk(np.hstack(self.pred), np.hstack(self.scores),
                                np.hstack(self.real))
        self.log_dict({**{k + '_test': v for k, v in metrics.items()}}, on_step=False, on_epoch=True, sync_dist=True)



    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=1e-4, weight_decay=10e-2)

    def train_dataloader(self):
        return train_ds

    def val_dataloader(self):
        return val_ds

    def test_dataloader(self):
        return test_ds



# ## Train the model

# Start tensorboard.


# Let's initialize the model, and train it!
#


name = "Task-"+str(task)+"_NFT-Timing-Whisper_"+ LG +'_'+dt[:2]+'_iter-' + str(iter)
arch = 'Emb'
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='UAR_val',
    patience=100,
    strict=False,
    verbose=False,
    mode='max'
)

checkpoint_callback = ModelCheckpoint(
    monitor="UAR_val",
    dirpath=r"./checkpoints/",
    filename=arch+'_'+name,
    mode="max",
)



from pytorch_lightning.loggers import WandbLogger

epochs = 500
lr = 1e-4


model = Whp_Ti(class_weights =class_weights )



trainer = Trainer(enable_model_summary=True, max_epochs=epochs, callbacks=[checkpoint_callback,early_stop_callback], 
                    accelerator="gpu", devices=2, gradient_clip_val=1)

trainer.fit(model)


gc.collect()
torch.cuda.empty_cache()
# tokenizer=processor)
