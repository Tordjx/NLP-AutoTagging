#%%
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
vocab = {}
def extract_data(result ,line, special_characters):
    #IMMONDE : mais j'ai pas trouvé mieux
    current_char = None
    current_word = ""
    for letter in line :
        if letter in special_characters: 
            if current_char != None : 
                result[current_char].append(current_word[:-1])
                if current_word[:-1] not in vocab.keys() :
                    vocab[current_word[:-1]] = len(vocab)
            current_char = letter
            current_word = ""
        else:
            current_word = current_word+letter
    return result
#AJOUTER UN TOKENIZEER PAR PITIE
nasty_data = {}
for c in special_chars:
    nasty_data[c] = []
for k in data : 
    line = data[k]
    line = line.replace("\n","")
    nasty_data = extract_data(nasty_data,line, special_chars)


# Replace keys in second_dict using the mapping
pretty_data = {special_char_dict[key]: value for key, value in nasty_data.items()}
pretty_data

# %%
import torch
from torch.utils.data import Dataset, DataLoader
class DictDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.classes = list(data_dict.keys())

    def __len__(self):
        return sum(len(self.data_dict[class_name]) for class_name in self.classes)

    def __getitem__(self, idx):
        class_idx = 0
        while idx >= len(self.data_dict[self.classes[class_idx]]):
            idx -= len(self.data_dict[self.classes[class_idx]])
            class_idx += 1
        class_name = self.classes[class_idx]
        class_id = pretty_names.index(class_name)
        return vocab[self.data_dict[class_name][idx]], class_id

# Create a dataset instance
full_dataset = DictDataset(pretty_data)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
batch_size = 64
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])# Create a DataLoader from the dataset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# %%
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self, num_emb,hsize, nclasses):
        super().__init__()
        self.emb = nn.Embedding(num_emb,hsize)
        self.fc = nn.Sequential(nn.Linear(hsize, hsize//2),nn.ReLU(),nn.Linear(hsize//2,nclasses) , nn.Softmax(1))
    def forward(self,x) :
        return self.fc(self.emb(x))
    def predict(self, x)  : 
        return torch.argmax(self.forward(x),-1)
    
model = Classifier(len(vocab), 256, len(special_chars))
# %%
opt = torch.optim.Adam
criterion = nn.CE

