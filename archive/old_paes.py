"""PAES model"""

import torch
import torch.nn as nn


class SoftAttention(nn.Module):

    def __init__(self, hidden_dim: int) -> None:
        super(SoftAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.tanh(self.w(h))

        weight = self.v(w)
        weight = weight.squeeze(dim=-1)

        weight = torch.softmax(weight, dim=1)
        weight = weight.unsqueeze(dim=-1)
        out = torch.mul(h, weight.repeat(1, 1, h.size(2)))

        out = torch.sum(out, dim=1)

        return out
    

class PAES(nn.Module):
    def __init__(
        self,
        max_num: int,
        max_len: int,
        linguistic_feature_size: int,
        readability_feature_size: int,
        pos_vocab: dict,
        embed_dim: int = 50,
        cnn_filters: int = 100,
        cnn_kernel_size: int = 5,
        lstm_units: int = 100,
        dropout: float = 0.5
    ) -> None:
        
        super(PAES, self).__init__()

        self.N = max_num
        self.L = max_len
        self.embed_dim = embed_dim

        self.embed_layer = nn.Embedding(
            num_embeddings=len(pos_vocab),
            embedding_dim=self.embed_dim,
            padding_idx=0
            )
        self.dropout = nn.Dropout(dropout)
        self.conv1d = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size
            )
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True
            )
        self.word_att = SoftAttention(lstm_units)
        self.sent_att = SoftAttention(lstm_units)
        self.linear = nn.Linear(lstm_units+linguistic_feature_size+readability_feature_size, 1)