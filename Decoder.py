import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import dataset
import numpy as np 
import matplotlib.pyplot as plt 

class CausalSelfAttention(nn.Module):
    def __init__(self , d_k ,d_model , n_heads , max_len):
        super().__init__()
            
        self.d_k = d_k
        self.n_heads = n_heads
        self.key = nn.Linear(d_model , d_k * n_heads)
        self.query = nn.Linear(d_model , d_k * n_heads)
        self.value = nn.Linear(d_model , d_k * n_heads)
        self.fc = nn.Linear(n_heads * d_k , d_model)

        cm = torch.tril(torch.ones(max_len , max_len))
        self.register_buffer('causal_mask' , cm.view(1 , 1 , max_len , max_len))
        
        def forward(self , q , k , v , pad_mask = None):
            q = self.query(q)
            v = self.value(v)
            k = self.key(k)
            
            N = q.shape[0]
            T = q.shape[1]
            
            q = q.view(N , T , self.n_heads , self.d_k).transpose(1 , 2)
            k = k.view(N , T , self.n_heads , self.d_k).transpose(1 , 2)
            v = v.view(N , T , self.n_heads , self.d_k).transpose(1 , 2)
            
            attn_scores = (q @ k.transpose(-2 , -1)) / math.sqrt(self.d_k)
            if pad_mask is not None:
                attn_scores = attn.masked_fill(self.causal_mask[: , None , None , :] == 0 , float('-inf'))
            attn_scores = attn_scores.masked_fill(pad_mask[:, :, :T, :T] == 0, float('-inf'))    
            attn_weights = F.softmax(attn_scores , dim = -1)
            
            A = attn_weights @ v
            A = A.transpose(1 , 2)
            A = A.contiguous().view(N , T , self.n_heads * self.d_k)
            
            return self.fc(A)
        
class TransformerBlock(nn.Module):  
    def __init__(self , d_model , d_k , n_heads , max_len , dropout_prob=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = CausalSelfAttention(d_k , d_model , n_heads , max_len)
        self.ann = nn.Sequential(
            nn.Linear(d_model , d_model * 4) ,
            nn.GELU() ,
            nn.Linear(d_model * 4 , d_model),
            nn.Dropout(dropout_prob),    )
        
        self.dropout = nn.Dropout(p = dropout_prob)
        
    def forward(self , x , pad_mask = None):
        x = self.ln1(x + self.mha(x , x , x , pad_mask))
        x = self.ln2(x + self.ann(x))
        
        return x
    
class Decoder(nn.Module):
    def __init__(self , vocab_size , d_model , d_k , n_heads , max_len , n_layers , dropout_prob):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size , d_model)
        self.pos_embedding = nn.Embedding(max_len , d_model)
        transformer_blocks = [
          TransformerBlock(d_model , d_k , n_heads , max_len , dropout_prob) for _ in range(n_layers)  
        ]
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model , vocab_size)
        
        def forward(self , x , pad_mask = None):
            x = self.embedding(x) 
            x = self.pos_embedding(x)
            for block in self.transformer_blocks:
                x = block(x , pad_mask)
            x = self.ln(x)
            x = self.fc(x)
            return x
        
model = Decoder(20_000 , 1024, 16 , 64, 4, 2, 0.1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
model.to(device) 

x = np.random.randint(0 , 20_000 , size=(8 , 512))
x_t = torch.tensor(x).to(device) 
y = model(x_t)
print(y.shape)

mask = np.ones((8 , 512 ))
mask[:, 256:] = 0
mask_t = torch.tensor(mask).to(device)

y = model(x_t , mask_t)
print(y.shape)

from transformers import AutoTokenizer , DataCalltorWithPadding
checkpoint = 'distilroberta-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

from datasets import load_dataset
raw_datasets = load_dataset('glue' , 'sst2')

def tokenize_fn(batch):
    return tokenizer(batch['sentence'] , truncation = True)
tokenized_datasets = raw_datasets.map(tokenize_fn , batched = True)
data_collator = DataCalltorWithPadding(tokenizer = tokenizer )
print(tokenized_datasets)

tokenized_datasets = tokenized_datasets.remove_columns(['sentence' , 'idx' , "label"])

from torch.utils.data import DataLoader
train_dataloader = DataLoader(tokenized_datasets['train'] , batch_size =32 , shuffle = True , collate_fn = data_collator)

for batch in train_dataloader:
    for k , v in batch.items():
        print(k , v.shape)
    break

print(tokenizer.pad_token_id)

model = Decoder(vocab_size=tokenizer.vocab_size,max_len=tokenizer.model_max_input_sizes[checkpoint], d_model=64, d_k=16, n_heads=4, n_layers=2, dropout_prob=0.1)
credits = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters())

from datetime import datetime

def train(model , criterion , optimizer , train_loader , epochs):
    train_losses = np.zeros(epochs)
    
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            targets = batch['input_ids'].clone().datach()
            targets[:, :-1] = tokenizer.pad_token_id
            
            outputs = model(batch['input_ids'] , batch['attention_mask'])
            loss = criterion(outputs.transpose(2,1), targets)
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        train_loss = np.mean(train_loss) 
        train_losses[it] = train_loss
        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs} - Train Loss: {train_loss:.4f} - Time: {dt}')
        
    return train_losses       

train_losses = train(model , criterion  , optimizer , train_dataloader , epochs =15)

vald_loader = DataLoader(tokenized_datasets['validation'] , batch_size =1 , collate_fn = data_collator)

model.eval()
for batch in vald_loader:
    #move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(batch['input_ids'] , batch['attention_mask'])
    break

print(outputs.shape)
print(torch.argmax(outputs , axis = -1))
prediction_ids = torch.argmax(outputs , axis = -1)
print(tokenizer.decode(prediction_ids[0]))
print(tokenizer.decode(batch['input_ids'][0]))
print(tokenizer.decode(batch['input_ids'][0,:5] , prediction_ids[0,:4]))

prompt = "it is "

tokenized_prompt = tokenizer(prompt , return_tensors = 'pt')
print(tokenized_prompt)

outputs = model(tokenized_prompt['input_ids'][:,:-1].to(device) , tokenized_prompt['attention_mask'][:,:-1].to(device))
print(outputs.shape)

prediction_ids = torch.argmax(outputs[:, :-1, :] , axis = -1)
print(tokenizer.decode(prediction_ids[0]))

prompt = "it is a"
tokenized_prompt = tokenizer(prompt , return_tensors = 'pt')
input_ids = tokenized_prompt['input_ids'][:,:-1].to(device)
attention_mask = tokenized_prompt['attention_mask'][:,:-1].to(device)

for _ in range(20):
    outputs = model(input_ids , attention_mask)
    prediction_ids = torch.argmax(outputs[:, -1, :] , axis = -1).unsqueeze(0)
    input_ids = torch.hastack(input_ids , prediction_ids.view(-1, 1))
    attention_mask = torch.ones_like(input_ids)
    
    if prediction_ids.item() == tokenizer.eos_token_id:
        break
    
tokenizer.decode(input_ids[0])    