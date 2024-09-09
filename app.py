import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request, url_for
entreno = True

# flask
app = Flask(__name__)
@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /query endpoint with a query to the 42BOT'})

# Configuration
embed_size = 384
block_size = 256
dropout = 0.2
n_layer = 6
n_head = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess text data
with open("autostop.txt", "r") as file:
    text = file.read()
preprocessed = list(text)

vocab = sorted(set(preprocessed))
vocab_size = len(vocab)

def encode(x):
    return [vocab.index(i) for i in x]

def decode(x):
    return ''.join([vocab[i] for i in x])

# Example usage
print(decode(encode("Respuesta ultima a la vida, el universo y todo lo demas")))

tokenized = torch.tensor(encode(preprocessed))
train_data = tokenized[:int(len(tokenized) * 0.9)]
val_data = tokenized[int(len(tokenized) * 0.9):]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (32,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch('train')

# Model components
class trans_block(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = Heads(heads, embed_size // heads)
        self.ff_layer = FF_Layer(embed_size)
        self.lnorm1 = nn.LayerNorm(embed_size)
        self.lnorm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attention(self.lnorm1(x))
        x = x + self.ff_layer(self.lnorm2(x))
        return x

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Batches, Time, Channels = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * Channels**-0.5
        wei = wei.masked_fill(self.tril[:Time, :Time] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class Heads(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.projection = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(out))

class FF_Layer(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        self.blocks = nn.Sequential(
            *[trans_block(embed_size, heads=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(embed_size)

    def forward(self, idx, targets=None):
        Branch, Time = idx.shape
        token_embed = self.embedding_table(idx)
        position_embed = self.position_embedding_table(torch.arange(Time, device=device))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            Batch, Time, Channel = logits.shape
            logits = logits.view(Batch * Time, Channel)
            targets = targets.view(Batch * Time)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        return logits, None

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_condition = idx[:, -block_size:]
            logits, _ = self(idx_condition)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
modeloalgo2 = BigramLM()
modelo2 = modeloalgo2.to(device)
print(entreno)
#if entreno == False:
    #modelo2.load_state_dict(torch.load("/content/drive/MyDrive/IA/autostop.txt"))
#else:
modelo2.load_state_dict(torch.load("entrenado.pt", map_location=torch.device("cpu")))

total_params = sum(p.numel() for p in modelo2.parameters())
print(f"params: {total_params}")
@app.route('/query', methods=['POST'])
def query():
    if request.method == 'POST':
        data = request.get_json()
        context = data['context']
        print(context)
        contextc = torch.tensor([encode(context)])
        context = contextc.to(device)
        return jsonify({'response' : decode(modelo2.generate(context,max_tokens = 250)[0].tolist())})
'''
#  data = request.get_json()
#  print(data)
context = request.data
    contextc = torch.tensor([encode(context)])
    context = contextc.to(device)
    return jsonify({'response' : decode(modelo2.generate(context,max_tokens = 500)[0].tolist())})


while True:
  contextc = input("que le dices?? (escribir SALIR para salir) ") # for hack club people, type SALIR to exit
  print("\n")
  contextc = torch.tensor([encode(contextc)])
  context = contextc.to(device)
  if(decode(context[0].tolist()) == "SALIR"):
    break
  print(decode(modelo2.generate(context,max_tokens = 500)[0].tolist()))
  print("\n")
'''
if __name__ == '__main__':
    app.run()
