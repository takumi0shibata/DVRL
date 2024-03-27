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
        self.cnn_filters = cnn_filters

        self.embed_layer = nn.Embedding(
            num_embeddings=len(pos_vocab),
            embedding_dim=self.embed_dim,
            padding_idx=0
            )
        self.dropout = nn.Dropout(dropout)
        self.conv1d = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.cnn_filters,
            kernel_size=cnn_kernel_size
            )
        self.lstm = nn.LSTM(
            input_size=self.cnn_filters,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True
            )
        self.word_att = SoftAttention(lstm_units)
        self.sent_att = SoftAttention(lstm_units)
        self.linear = nn.Linear(lstm_units+linguistic_feature_size+readability_feature_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        linguistic_features: torch.Tensor,
        readability_features: torch.Tensor
    ) -> torch.Tensor:
        
        embed = self.embed_layer(x)
        embed = self.dropout(embed)
        embed = embed.view(-1, self.L, self.embed_dim)  # size: (バッチサイズ * N, L, embed_dim)
        sentence_cnn = self.conv1d(embed.permute(0, 2, 1))
        sentence_att = self.word_att(sentence_cnn.permute(0, 2, 1))
        
        # 文ごとの結果を再構成
        sentence_fea = sentence_att.view(-1, self.N, self.cnn_filters)  # size: (バッチサイズ, N, 100)
        
        essay_lstm, _ = self.lstm(sentence_fea)
        essay_fea = self.sent_att(essay_lstm)
        essay_fea = torch.cat([essay_fea, linguistic_features, readability_features], dim=1)
        essay_fea = self.linear(essay_fea)
        essay_fea = torch.sigmoid(essay_fea)
        
        return essay_fea
    
    # # CNN-CNN version
    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     linguistic_features: torch.Tensor,
    #     readability_features: torch.Tensor
    # ) -> torch.Tensor:
        
    #     embed = self.embed_layer(x)
    #     embed = self.dropout(embed)
    #     embed = embed.view(-1, self.L, self.embed_dim)  # size: (バッチサイズ * N, L, embed_dim)
    #     sentence_cnn = self.conv1d(embed.permute(0, 2, 1))
    #     sentence_att = self.word_att(sentence_cnn.permute(0, 2, 1))
        
    #     # 文ごとの結果を再構成
    #     sentence_fea = sentence_att.view(-1, self.N, self.cnn_filters)  # size: (バッチサイズ, N, 100)
        
    #     essay_fea = self.conv1d2(sentence_fea.permute(0, 2, 1))
    #     essay_fea = self.sent_att(essay_fea.permute(0, 2, 1))
    #     essay_fea = torch.cat([essay_fea, linguistic_features, readability_features], dim=1)
    #     essay_fea = self.linear(essay_fea)
    #     essay_fea = torch.sigmoid(essay_fea)
        
    #     return essay_fea
    

# PAES model with tiny version (based on Taghipour and Ng, 2016 model)
class tinyPAES(PAES):
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
        
        super(tinyPAES, self).__init__(
            max_num=max_num,
            max_len=max_len,
            linguistic_feature_size=linguistic_feature_size,
            readability_feature_size=readability_feature_size,
            pos_vocab=pos_vocab,
            embed_dim=embed_dim,
            cnn_filters=cnn_filters,
            cnn_kernel_size=cnn_kernel_size,
            lstm_units=lstm_units,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        linguistic_features: torch.Tensor,
        readability_features: torch.Tensor
    ) -> torch.Tensor:
        
        embed = self.embed_layer(x)
        embed = self.dropout(embed) # size: (batch_size, num_token, embed_dim)
        cnn_output = self.conv1d(embed.permute(0, 2, 1)) # size: (batch_size, cnn_filters, num_token)
        
        essay_fea, _ = self.lstm(cnn_output.permute(0, 2, 1)) # size: (batch_size, num_token, lstm_units)
        essay_fea = torch.mean(essay_fea, dim=1, keepdim=False) # size: (batch_size, lstm_units)
        essay_fea = torch.cat([essay_fea, linguistic_features, readability_features], dim=1)
        essay_fea = self.linear(essay_fea)
        essay_fea = torch.sigmoid(essay_fea)
        
        return essay_fea
