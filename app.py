import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import requests
import os

# Configuration
embed_size = 384
block_size = 256
dropout = 0.2
n_layer = 6
n_head = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess text data
with open("autostop.txt", "r") as file:
file_url = "https://filebin.net/kws1ohude8xyphjp/autostop.txt"
file_path = "autostop.txt"

if not os.path.isfile(file_path):
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(file_path, "w") as file:
            file.write(response.text)

with open(file_path, "r") as file:
    text = file.read()
preprocessed = list(text)

@@ -25,9 +34,6 @@ def encode(x):
def decode(x):
    return ''.join([vocab[i] for i in x])

# Example usage
print(decode(encode("Respuesta ultima a la vida, el universo y todo lo demas")))

tokenized = torch.tensor(encode(preprocessed))
train_data = tokenized[:int(len(tokenized) * 0.9)]
val_data = tokenized[int(len(tokenized) * 0.9):]
@@ -41,7 +47,6 @@ def get_batch(split):

xb, yb = get_batch('train')

# Model components
class trans_block(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
@@ -140,12 +145,15 @@ def generate(self, idx, max_tokens):

modeloalgo2 = BigramLM()
modelo2 = modeloalgo2.to(device)
modelo2.load_state_dict(torch.load("entrenado.pt", map_location=torch.device("cpu")))

try:
    modelo2.load_state_dict(torch.load("entrenado.pt", map_location=torch.device("cpu")))
except FileNotFoundError:
    st.error("Model file not found. Initializing with random weights.")

total_params = sum(p.numel() for p in modelo2.parameters())
print(f"params: {total_params}")
st.write(f"Model parameters: {total_params}")

# Streamlit app
st.title('Chatbot')
st.write("Type your query below:")