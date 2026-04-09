import torch
import torch.nn as nn
import torch.optim as optim
import math

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -----------------------------
# Multi Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.fc(out)


# -----------------------------
# Feed Forward
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Transformer Block
# -----------------------------
class Block(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model, 128)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


# -----------------------------
# Final Model
# -----------------------------
class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, 64)
        self.pos = PositionalEncoding(64)

        self.blocks = nn.Sequential(
            Block(64, 8),
            Block(64, 8)
        )

        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        x = self.blocks(x)

        x = x.mean(dim=1)
        return self.fc(x)


# -----------------------------
# DATASET (Simple Sentiment)
# -----------------------------
sentences = [
    "i love this",
    "this is great",
    "i hate this",
    "this is bad"
]

labels = [1, 1, 0, 0]   # 1 = positive, 0 = negative

# Build vocab
vocab = {}
idx = 0
for s in sentences:
    for w in s.split():
        if w not in vocab:
            vocab[w] = idx
            idx += 1

# Convert text to numbers
max_len = 4
data = []

for s in sentences:
    tokens = [vocab[w] for w in s.split()]
    tokens += [0]*(max_len - len(tokens))
    data.append(tokens)

X = torch.tensor(data)
y = torch.tensor(labels)

# -----------------------------
# TRAINING
# -----------------------------
model = Model(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -----------------------------
# TEST
# -----------------------------
# -----------------------------
# TEST WITH TEXT OUTPUT
# -----------------------------
test_sentence = "i hate this"

# Convert text → tokens
tokens = [vocab[word] for word in test_sentence.split()]
tokens += [0] * (max_len - len(tokens))

test_tensor = torch.tensor([tokens])

# Prediction
pred = model(test_tensor)
pred_class = torch.argmax(pred, dim=1).item()

# Print result
print("\n===== FINAL OUTPUT =====")
print("Input Sentence:", test_sentence)

if pred_class == 1:
    print("Predicted Sentiment: Positive ")
else:
    print("Predicted Sentiment: Negative ")