#%%
from bpe import Encoder
import json
import pandas as pd


with open('entities.json', 'r') as json_file:
    data = json.load(json_file)
import yaml
with open('tokens.yml', 'r') as file:
    tokens = yaml.safe_load(file)
special_char_dict = {}
for key, value in tokens.items():
    special_char_dict[value['start']] = key
special_chars = list(special_char_dict.keys())
pretty_names = list(special_char_dict.values())
good_data= []
import re
def extract_data(result ,line, special_characters):
    current_char = None
    current_word = ""
    for letter in line :
        if letter in special_characters: 
            if current_char != None : 
                result[current_char].append(current_word[:-1])
            current_char = letter
            current_word = ""
        else:
            current_word = current_word+letter
    return result
nasty_data = {}
for c in special_chars:
    nasty_data[c] = []
for k in data : 
    line = data[k]
    line = line.replace("\n","")
    nasty_data = extract_data(nasty_data,line, special_chars)


# Replace keys in second_dict using the mapping
pretty_data = {special_char_dict[key]: value for key, value in nasty_data.items()}

#%%
bpe_data = []
for k in pretty_data.keys() :
    bpe_data = bpe_data+ pretty_data[k]
from bpe import Encoder
encoder = Encoder()
encoder.fit(bpe_data)
tokenized_data = {key: list(encoder.transform(value)) for key, value in pretty_data.items()}
#%%

#%%
import torch
from torch.utils.data import Dataset, DataLoader
class DataFromDict(Dataset):
    def __init__(self,input_dict ):
        self.input_dict = input_dict
        data = []
        self.len = 0
        for k in list(input_dict.keys()):
            self.len +=len(input_dict[k])
            print(len(input_dict[k]))
            for v in input_dict[k]:
                data.append((v,list(input_dict.keys()).index(k)))
        self.data = data
    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        x,y = self.data[idx]
        return torch.Tensor(x),torch.Tensor([y])
full_dataset = DataFromDict(tokenized_data)
print(len(full_dataset))
# %%

from torch.nn.utils.rnn import pad_sequence
class PadSequence:
    def __call__(self, batch):
        (xx, yy) = zip(*batch)
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        return xx_pad.to(torch.int), torch.cat(yy).to(torch.int)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
batch_size = 512
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])# Create a DataLoader from the dataset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=PadSequence(), num_workers = 15)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=PadSequence(),num_workers = 15)

next(iter(train_dataloader))
# %%
import lightning as L
import torch.nn as nn
from torch import Tensor
import math
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 20):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
from torchmetrics import Accuracy, Recall, F1Score
class Classifier(L.LightningModule):
    def __init__(self, num_emb,hsize, nclasses,dropout =0.1):
        super().__init__()
        self.posenc = PositionalEncoding(emb_size=hsize,dropout=dropout)
        self.emb = nn.Embedding(num_emb,hsize)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hsize, nhead=4, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(hsize, hsize//2),nn.ReLU(),nn.Linear(hsize//2,nclasses) , nn.Softmax(1))
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.05)
        self.accuracy = Accuracy(task="multiclass", num_classes=nclasses, average = "weighted")
        self.recall = Recall(task="multiclass", num_classes=nclasses, average = "weighted")
        self.f1score = F1Score(task="multiclass", num_classes=nclasses, average = "weighted")
    def forward(self,x) :
        pad_mask = (x== 0).transpose(0, 1)
        emb = self.emb(x)
        enc = self.transformer_encoder(emb,src_key_padding_mask = pad_mask)
        enc = enc.mean(1)
        return self.fc(enc)
    def predict(self, x)  : 
        return torch.argmax(self.forward(x),-1)
    def training_step(self, batch):
        x,y = batch
        y = y.to(torch.long)
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        self.log("train_loss", loss, prog_bar = True)

        return loss
    def validation_step(self,batch) :
        self.eval()
        x,y = batch
        y = y.to(torch.long)
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        self.log("val_loss", loss, prog_bar = True)
        self.train()
        predictions = torch.argmax(x_hat,-1)
        self.log("accuracy", self.accuracy(predictions,y), prog_bar = True)
        self.log("recall", self.recall(predictions,y), prog_bar = True)
        self.log("f1score", self.f1score(predictions,y), prog_bar = True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
model = Classifier(encoder.vocab_size, 256, len(special_chars))
for name, param in model.named_parameters():
    if 'weight' in name and param.data.dim() == 2:
        nn.init.kaiming_uniform_(param)
# %%
trainer = L.Trainer(max_epochs=30, devices = 1,log_every_n_steps=1)
trainer.fit(model, train_dataloader, test_dataloader)

# %%

trainer.validate(model, test_dataloader)

#%%
from transformers import pipeline
model_name = "facebook/bart-large-mnli" #beaucoup beaucoup mieux !!!!!!!
#model_name = "mtheo/camembert-base-xnli"
classifier = pipeline("zero-shot-classification", 
                      model=model_name, device = 0)

candidate_labels = ['age',
 'date de naissance',
 'etat civil',
 "niveau d'education",
 'employeur',
 'prenom',
 'lien',
 'travail',
 'nom de jeune fille',
 'nationalité',
 'observation',
 'profession',
 'nom',
 'nom du ménage']
preds,tgt = [],[]
from tqdm import tqdm
for x, y in tqdm(test_dataset) : 
    x = x.numpy()
    x = next(encoder.inverse_transform([x]))
    if x =="" :
        continue
    preds.append(classifier(x, candidate_labels)['labels'][0])
    tgt.append(y)
#%%
preds = [candidate_labels.index(u) for u in preds]
# %%

tgt, preds = torch.Tensor(tgt),torch.Tensor(preds)
accuracy = Accuracy(task="multiclass", num_classes=len(pretty_names), average = "weighted")(tgt, preds)
recall = Recall(task="multiclass", num_classes=len(pretty_names), average = "weighted")(tgt, preds)
f1 = F1Score(task="multiclass", num_classes=len(pretty_names), average = "weighted")(tgt, preds)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-score:", f1)
# %%
