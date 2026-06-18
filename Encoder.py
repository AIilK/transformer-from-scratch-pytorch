import math
import torch 
import torch.nn as nn
import torch.nn.functional as f 
from torch.utils.data import dataset 
import numpy as np 
import matplotlib.pyplot as plt 

class MultuHeadAttention(nn.Module):
    def __init__(self, d_k,d_model, n_heads):
        super(MultuHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
                
        assert d_k * n_heads == d_model        
        
        self.q_linear = nn.Linear(d_model, d_k * n_heads)
        self.k_linear = nn.Linear(d_model, d_k * n_heads)
        self.v_linear = nn.Linear(d_model, d_k * n_heads)
        
        self.fc = nn.Linear(d_k * n_heads, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        # Linear projections
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        
        N = q.shape[0]
        T = q.shape[1]
        
        q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        # formula
        attn_score = q @ k.transpose(-2,-1) / math.sqrt (self.d_k)
        #mask
        if mask is not None :
            attn_score = attn_score.masked_fill(
                mask[: , None , None , :] == 0, float('-inf'))
        #softmax    
        attn_weights = f.softmax(attn_score , dim = -1)    
        # mut with V
        A = attn_weights @ v
        # Integration head
        A = A.transpose(1,2)
        A = A . contiguous().view (N , T , self.d_k * self.n_heads)    
     
        return self.fc(A)
    
class TransformerBlock (nn.Module):
    def __init__ (self , d_k , d_model , n_heads , dropout_prob = 0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultuHeadAttention(d_k , d_model , n_heads)
        self.ann = nn.Sequential(
            nn.Linear(d_model , d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4 , d_model),
            nn.Dropout(dropout_prob),
        )   
        self.dropout = nn.Dropout (p=dropout_prob)
        
    def forward (self , x, mask = None):
        x = self.ln1(x + self.mha(x,x,x, mask))
        x = self.ln2(x + self.ann(x))
        x = self.dropout (x)
        
        return x    
    
class PositionalEncodeing (nn.Module):
    def __init__(self , d_model , max_len = 2048 , dropout_prob = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout_prob)
            
        position = torch.arange (max_len).unssqueeze(1)
        exp_term = torch.arange(0 , d_model , 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len , d_model)
        pe[0, : , 0::2] = torch.sin (position * div_term)
        pe[0,: , 1::2] = torch.cos(position * div_term)
        self.register_buffer ('pe' , pe)
            
    def forward (self , x):
        x = x + self.pe[: , :x.size(1), :] 
        return self.dropout(x)          
    
class Encoder (nn.Module):
    def __init__ (self , vocab_size,max_len , d_k, d_model , n_heads, n_layers , n_classes , dropout_prob ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size , d_model) # -> (5 , 64)
        self.pos_encoding = PositionalEncodeing(d_model , max_len , dropout_prob)
        transformer_blocks = [
            TransformerBlock(d_k , d_model , n_heads , dropout_prob) for _ in range(n_layers)
        ]
        self.transformer_blocks =  nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm (d_model)
        self.fc = nn.Linear(d_model , n_classes)
        
    def forward (self,x,mask = None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks :
            x = block (x, mask)
            
        x = x [: , 0 , :]
        x = self.ln(x)
        x = self.fc(x)
        
        return x
        
model = Encoder (20_000 , 1024 , 16 , 64 , 4 , 2 , 5 , 0.1 )    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)         
model.to(device)

x = np.random.randit(0,20_00 , size = (5 , 512))
x_t = torch.tensor(x).to(device)

mask = np.ones((8,512))
mask[: , 256 :] = 0
mask_t = torch.tensor(mask).to(device)
y = model(x_t , mask_t)
print(y.shape)

# train and validation 

from transformers import AutoTokenizer , DataCollatorWthpadding 
from datasets import load_dataset

checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_datasets = load_dataset("glue" , "sst2")
print(raw_datasets)

def tokenize_fn (batch):
    return tokenizer (batch ['sentence'] , truncation = True)

tokenized_datasets = raw_datasets.map(tokenize_fn , batched = True)
data_collator = DataCollatorWthpadding(tokenizer = tokenizer)
print(data_collator)
print(tokenized_datasets)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence" , "idx"])
tokenized_datasets = tokenized_datasets.rename_columns(["label" , "labels"])
print(tokenized_datasets)

from torch.utils.data import DataLoader 
train_loader = DataLoader(
    tokenized_datasets ["train"] ,
    shuffle=True,
    batch_size = 32,
    collate_fn=data_collator
)

for batch in train_loader :
    for k,v in batch.items():
        print("k:" , k , "v:" , v.shape)
    break

set (tokenized_datasets['train']['labels'])
print(tokenizer.vocab_size)
print(tokenizer.max_model_input_size)

model = Encoder (vocab_size = tokenizer.vocab_size , max_len = tokenizer.max_model_input_size[checkpoint] , d_k = 16 , d_model = 64 , n_heads = 4 ,
                 n_layers = 2 , n_classes = 2 , dropout_prob = 0.1 )   
model.to(device) 
# loss and optimizer
creiterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr = 1e-4)

from datetime import datetime

def train (model , train_loader , creiterion , optimizer , valid_loader  , n_epochs = 5):
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)
    
    for it in range (n_epochs):
        model.train()
        t0 = datetime.now()
        train_losses = 0
        n_train = 0
        
        for batch in train_loader :
            batch = {k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad()
            output = model(batch['input_ids'] , batch['attention_mask'])
            loss = creiterion (output , batch['labels'])
            loss.backward()
            optimizer.step()
            train_losses += loss.item() * batch['input_ids'].shape[0]
            n_train += batch['input_ids'].shape[0]
            train_loss = train_losses / n_train
            
            model.eval()
            test_losses = 0
            n_test = 0
            for batch in valid_loader :
                batch = {k:v.to(device) for k,v in batch.items()}
                output = model(batch['input_ids'] , batch['attention_mask'])
                loss = creiterion (output , batch['labels'])
                test_losses += loss.item() * batch['input_ids'].shape[0]
                n_test += batch['input_ids'].shape[0]
            test_loss = test_losses / n_test
            
            train_losses[it] = train_loss
            test_losses[it] = test_loss
            
            dt = datetime.now() - t0
            print (f"Epoch {it+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Time: {dt}")
        return train_losses , test_losses
    
train_losses , test_losses = train (model , train_loader , creiterion , optimizer , valid_loader , n_epochs = 4)        

#accuracy
model.eval()
correct = 0
total = 0
for batch in valid_loader :
    batch = {k:v.to(device) for k,v in batch.items()}
    output = model(batch['input_ids'] , batch['attention_mask'])
    _, predicted = torch.max(output.data, 1)
    total += batch['labels'].size(0)
    correct += (predicted == batch['labels']).sum().item()
    train_acc = correct / total
    correct = 0
    total = 0
    for batch in valid_loader :
        batch = {k:v.to(device) for k,v in batch.items()}
        output = model(batch['input_ids'] , batch['attention_mask'])
        _, predicted = torch.max(output.data, 1)
        total += batch['labels'].size(0)
        correct += (predicted == batch['labels']).sum().item()
test_acc = correct / total
print(f"Train Accuracy: {train_acc:.4f} - Test Accuracy: {test_acc:.4f}")        