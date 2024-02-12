import torch
import torch.nn as nn
from custom_layers.soft_attention import SoftAttention


class PAES(nn.Module):
    def __init__(self, max_num, max_len, linguistic_feature_size, readability_feature_size, embed_dim=50, pos_vocab=None):
        super(PAES, self).__init__()
        self.N = max_num
        self.L = max_len
        self.embed_dim = embed_dim

        self.embed_layer = nn.Embedding(num_embeddings=len(pos_vocab), embedding_dim=embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=5)
        self.lstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True)
        self.word_att = SoftAttention(100)
        self.sent_att = SoftAttention(100)
        self.linear = nn.Linear(100+linguistic_feature_size+readability_feature_size, 1)

    def forward(self, x, linguistic_features, readability_features):
        embed = self.embed_layer(x)
        embed = nn.Dropout(0.5)(embed)
        embed = embed.view(embed.size()[0], self.N, self.L, self.embed_dim)
        sentence_fea = torch.tensor([], requires_grad=True).to(x.device)
        for n in range(self.N):
            sentence_embed = embed[:, n, :, :]
            sentence_cnn = self.conv1d(sentence_embed.permute(0, 2, 1))
            sentence_att = self.word_att(sentence_cnn.permute(0, 2, 1))
            sentence_fea = torch.cat([sentence_fea, sentence_att.unsqueeze(1)], dim=1)

        essay_lstm, _ = self.lstm(sentence_fea)
        essay_fea = self.sent_att(essay_lstm)
        essay_fea = torch.cat([essay_fea, linguistic_features, readability_features], dim=1)
        essay_fea = self.linear(essay_fea)
        essay_fea = torch.sigmoid(essay_fea)
        
        return essay_fea