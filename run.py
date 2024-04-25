import json
import pandas as pd
import yaml
import re
from torch import Tensor
from tqdm import tqdm
from unidecode import unidecode
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math
from torchmetrics import Accuracy, Recall, F1Score
import lightning as L
from transformers import pipeline
from bpe import Encoder

# Load entities data from JSON file
with open('entities.json', 'r') as json_file:
    entities_data = json.load(json_file)

# Load tokens data from YAML file
with open('tokens.yml', 'r') as file:
    tokens = yaml.safe_load(file)

# Create a dictionary to map special characters to their respective tokens
special_char_dict = {value['start']: key for key, value in tokens.items()}
special_chars = list(special_char_dict.keys())

# Normalize function to remove accents and convert to lowercase
def normalize(word):
    word = word.replace(' ', "")
    text_cleaned = unidecode(word)
    text_lower = text_cleaned.lower()
    return text_lower

# Function to extract data from each line based on special characters
def extract_data(result, line, special_characters):
    current_char = None
    memory = {k: None for k in special_characters}
    current_word = ""
    for letter in line:
        if letter in special_characters:
            if current_char is not None and normalize(current_word) != "neant":
                current_word = current_word.replace(' ', '')
                if normalize(normalize(current_word)) == "idem":
                    current_word = memory[letter]
                memory[letter] = current_word
                if current_word is not None and current_word != "":
                    result[current_char].append(current_word)
            current_char = letter
            current_word = ""
        else:
            current_word = current_word + letter
    return result

# Initialize a dictionary to store extracted data
extracted_data = {c: [] for c in special_chars}

# Extract data from entities data
for key, line in entities_data.items():
    line = line.replace("\n", "")
    extracted_data = extract_data(extracted_data, line, special_chars)

# Map keys in the extracted data dictionary using the special character dictionary
mapped_data = {special_char_dict[key]: value for key, value in extracted_data.items()}

# Initialize a list to store all the data for BPE encoding
bpe_data = []
for k in mapped_data.keys():
    bpe_data.extend(mapped_data[k])

# Perform BPE encoding
encoder = Encoder()
encoder.fit(bpe_data)
tokenized_data = {key: list(encoder.transform(value)) for key, value in mapped_data.items()}

# Define a custom Dataset class for PyTorch
class DataFromDict(Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict
        data = []
        self.len = 0
        for k in list(input_dict.keys()):
            self.len += len(input_dict[k])
            for v in input_dict[k]:
                data.append((v, list(input_dict.keys()).index(k)))
        self.data = data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.Tensor(x), torch.Tensor([y])

# Create datasets from tokenized data
full_dataset = DataFromDict(tokenized_data)

# Split dataset into train and test sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
batch_size = 512
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
class PadSequence:
    def __call__(self, batch):
        (xx, yy) = zip(*batch)
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        return xx_pad.to(torch.int), torch.cat(yy).to(torch.int)

# Create DataLoaders from datasets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=15)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=PadSequence(), num_workers=15)

# Define a positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 20):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# Define a classifier model
class Classifier(L.LightningModule):
    def __init__(self, num_emb, hsize, nclasses, dropout=0.1):
        super().__init__()
        self.posenc = PositionalEncoding(emb_size=hsize, dropout=dropout)
        self.emb = nn.Embedding(num_emb, hsize)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hsize, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(hsize, hsize // 2), nn.ReLU(), nn.Linear(hsize // 2, nclasses), nn.Softmax(1))
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        self.accuracy = Accuracy(task="multiclass", num_classes=nclasses, average="weighted")
        self.recall = Recall(task="multiclass", num_classes=nclasses, average="weighted")
        self.f1score = F1Score(task="multiclass", num_classes=nclasses, average="weighted")

    def forward(self, x):
        pad_mask = (x == 0).transpose(0, 1)
        emb = self.emb(x)
        enc = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)
        enc = enc.mean(1)
        return self.fc(enc)

    def predict(self, x):
        return torch.argmax(self.forward(x), -1)

    def training_step(self, batch):
        x, y = batch
        y = y.to(torch.long)
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        self.eval()
        x, y = batch
        y = y.to(torch.long)
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.train()
        predictions = torch.argmax(x_hat, -1)
        self.log("accuracy", self.accuracy(predictions, y), prog_bar=True)
        self.log("recall", self.recall(predictions, y), prog_bar=True)
        self.log("f1score", self.f1score(predictions, y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

# Initialize the model
model = Classifier(encoder.vocab_size, 256, len(special_chars))

# Initialize weights using Kaiming Uniform initialization
for name, param in model.named_parameters():
    if 'weight' in name and param.data.dim() == 2:
        nn.init.kaiming_uniform_(param)

# Initialize Lightning Trainer
trainer = L.Trainer(max_epochs=30, devices=1, log_every_n_steps=1)

# Train the model
#trainer.fit(model, train_dataloader, test_dataloader)

# Validate the model
trainer.validate(model, test_dataloader)

# Initialize zero-shot classification pipeline using a pre-trained model
model_name = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"

classifier = pipeline("zero-shot-classification", model=model_name,  device_map="auto", batch_size=512)

# Define candidate labels
candidate_labels = ['age', 'birth date', 'civil status', "education level", 'employer', 'first name', 'parental link',
                    'place of birth', 'maiden name', 'nationality', 'observation', 'occupation', 'surname',
                    'household name']
"""candidate_labels = ['age', 'date de naissance', 'status marital', "niveau d'éducation", 'employeur', 'prénom', 'lien de parenté',
                    'lieu de naissance', 'nom de jeune fille', 'nationalit"y"', 'observation', 'travail', 'nom de famille',
                    'nom du ménage']"""
preds, tgt = [], []

# Perform zero-shot classification on test data
for x, y in tqdm(test_dataloader):
    x = x.numpy()
    x = [next(encoder.inverse_transform([u])).replace('__unk', "").replace(' ', '') for u in x]
    out = classifier(x, candidate_labels)
    for u in out:
        preds.append(u['labels'][0])
    tgt = tgt + [u.item() for u in y]

# Convert predictions and ground truth to indices
preds = [candidate_labels.index(u) for u in preds]
tgt, preds = torch.Tensor(tgt), torch.Tensor(preds)

# Compute metrics
accuracy = Accuracy(task="multiclass", num_classes=len(special_chars), average="weighted")(tgt, preds)
recall = Recall(task="multiclass", num_classes=len(special_chars), average="weighted")(tgt, preds)
f1 = F1Score(task="multiclass", num_classes=len(special_chars), average="weighted")(tgt, preds)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-score:", f1)

