#!/usr/bin/env python
# coding: utf-8

# Train simple Large Language Model (LLM)


import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
import tiktoken



# -----------------------------
# 1. Byte Pair Encoding (BPE) Tokenizer
# -----------------------------
class BPEDataset(Dataset):
    def __init__(self, filepath, seq_len=256):
        # Use GPT-2 BPE tokenizer from tiktoken
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.seq_len = seq_len

        # Read dataset
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Encode entire text to integer token IDs
        self.data = self.tokenizer.encode(text)
        self.vocab_size = self.tokenizer.n_vocab

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y


# -----------------------------
# 2. Simple Transformer model
# -----------------------------
class ToyGPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.token_emb(x)  # [batch, seq_len, hidden]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, hidden]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # back to [batch, seq_len, hidden]
        logits = self.head(x)
        return logits



# -----------------------------
# 3. Generate text from prompt
# -----------------------------
@torch.no_grad()
def generate_text(model, tokenizer, prompt="Hello", device="cpu", length=20, temperature=0.8):
    model.eval()

    # Encode the prompt → tensor of token IDs
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated = input_ids.clone()

    for _ in range(length):
        logits = model(generated[:, -model.seq_len:])  # forward pass last context
        next_token_logits = logits[0, -1] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_id.unsqueeze(0)), dim=1)

    # Decode to human-readable text
    text = tokenizer.decode(generated[0].tolist())
    return text




# -----------------------------
# 4. Chat with trained model (Inference)
# -----------------------------
def chat(prompt):
    if os.path.exists("toy_gpt_weights.pth"):
        model.load_state_dict(torch.load("toy_gpt_weights.pth", map_location=device, weights_only=True))
        model.eval()  # sets model to evaluation mode
        print("Generated text:", generate_text(model, dataset.tokenizer, prompt, length=20, device=device, temperature=1))
    else:
        print("Model not trained.")




# -----------------------------
# Model & Training Parameters
# -----------------------------

seq_len = 32        # Number of tokens the model sees at once (context window size)
batch_size = 128     # Number of sequences processed in parallel before each weight update
hidden_size = 4   # Dimensionality of token embeddings and hidden layers
num_layers = 1       # Number of stacked Transformer encoder layers
num_heads = 2        # Number of self-attention heads per Transformer layer
lr = 5e-4            # Learning rate for the optimizer (Adam)
epochs = 5           # Number of full passes over the dataset
temperature = 0.8    # reduces randomness → more readable output
num_workers = 0     # number of parallel data-loading subprocesses (should match number of CPU-cores)




# -----------------------------
# Dataset Preparation
# -----------------------------
print("Tokenize dataset")
dataset = BPEDataset("shakespeare_small.txt", seq_len)
vocab_size = dataset.vocab_size
print("Vocab size:", dataset.vocab_size)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToyGPT(vocab_size, hidden_size, num_layers, num_heads).to(device)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))




# -----------------------------
# Training loop
# -----------------------------
for epoch in range(epochs):
        torch.cuda.empty_cache()
        epoch_start = time.time()
        total_loss = 0.0
        for i, (x, y) in enumerate(loader):
            #data_start = time.time()
            x, y = x.to(device), y.to(device)
            #print(f"Batch transfer time: {time.time() - data_start:.4f}s")
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                       
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"Batch {i}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
        
        elapsed_time =  time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}, Time: {elapsed_time:.1f} s")
        print(f"\nGenerated text: {generate_text(model, dataset.tokenizer, prompt='clouds', device=device, temperature=0.8)}\n")




# -----------------------------
# Save weights
# -----------------------------
torch.save(model.state_dict(), "toy_gpt_weights.pth")
print("Model weights saved!")



chat("Coluds ")
