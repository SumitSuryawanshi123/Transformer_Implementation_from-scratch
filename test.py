import torch
import torch.nn as nn
import math

# Define the PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector of shape (seq_len, 1) for positions
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

# Example usage
d_model = 512  # Embedding dimension
seq_len = 100  # Maximum sequence length
dropout = 0.1  # Dropout rate

# Instantiate the PositionalEncoding class
pos_encoder = PositionalEncoding(d_model, seq_len, dropout)

# Create a batch of token embeddings (batch_size=32, seq_len=50, d_model=512)
batch_size = 32
sequence_length = 50
embeddings = torch.randn(batch_size, sequence_length, d_model)

# Apply positional encoding
encoded_embeddings = pos_encoder(embeddings)
print(embeddings)  # Output: (32, 50, 512)
