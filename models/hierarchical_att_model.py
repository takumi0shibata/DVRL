import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers.attention import AttentionTorch

class PAES_on_torch(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, pos_vocab_size, config, linguistic_feature_size, readability_feature_size, device):
        super(PAES_on_torch, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttNet(pos_vocab_size, config)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, linguistic_feature_size, readability_feature_size)
        self.device = device


    def forward(self, x_input, linguistic_features, readability_features):
        # 空のテンソルを作成
        output_list = torch.empty((0,), requires_grad=True).to(x_input.device)
        x_input = x_input.permute(1, 0, 2) # [sentnum, batch, sentlen]
        for seq in x_input:
            output = self.word_att_net(seq) # seq: [batch, sentlen]
            output_list = torch.cat((output_list, output)) # shape: [sentnum, batch, embedding_dim]
        output = self.sent_att_net(output_list, linguistic_features, readability_features)
        return output


class WordAttNet(nn.Module):
    def __init__(self, pos_vocab_size, config):
        super(WordAttNet, self).__init__()
        cnn_filters = config.CNN_FILTERS
        cnn_kernel_size = config.CNN_KERNEL_SIZE
        pos_embedding_dim = config.EMBEDDING_DIM
        dropout_prob = config.DROPOUT

        self.lookup = nn.Embedding(num_embeddings=pos_vocab_size, embedding_dim=pos_embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=pos_embedding_dim, out_channels=cnn_filters, kernel_size=cnn_kernel_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(cnn_filters, cnn_filters)
        self.fc2 = nn.Linear(cnn_filters, 1, bias=False)
        self.attention = AttentionTorch(cnn_filters, op='attsum', activation='tanh', init_stdev=0.01)


    def forward(self, inputs): # inputs: [batch, sentlen]
        output = self.lookup(inputs) # shape: [batch, sentlen, embedding_dim]
        output = self.dropout(output) # shape: [batch, sentlen, embedding_dim]
        output = output.permute(0, 2, 1) # shape: [batch, embedding_dim, sentlen] 
        output = self.conv1(output) # shape: [batch, embedding_dim, seq_len-4]
        output = output.permute(0, 2, 1) # shape: [batch, seq_len-4, embedding_dim]
        output = self.attention(output).unsqueeze(0) # shape: [1, batch, embedding_dim]
        return output


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size, word_hidden_size, linguistic_feature_size, readability_feature_size):
        super(SentAttNet, self).__init__()
        self.LSTM = nn.LSTM(word_hidden_size, sent_hidden_size)
        self.fc = nn.Linear(sent_hidden_size+linguistic_feature_size+readability_feature_size, 1)
        self.fc1 = nn.Linear(sent_hidden_size, sent_hidden_size)
        self.fc2 = nn.Linear(sent_hidden_size, 1, bias=False)
        self.attention = AttentionTorch(sent_hidden_size, op='attsum', activation='tanh', init_stdev=0.01)

    def forward(self, input, linguistic_features, readability_features): # input: [sentnum, batch, embedding_dim]
        f_output, _ = self.LSTM(input) # f_output: [sentnum, batch, sent_hidden_size]
        output = f_output.permute(1, 0, 2) # output: [batch, sentnum, sent_hidden_size]
        output = self.attention(output) # output: [batch, sent_hidden_size]
        output = torch.cat((output, linguistic_features, readability_features), 1) # output: [batch, sent_hidden_size+linguistic_feature_size+readability_feature_size]
        output = torch.sigmoid(self.fc(output)) # output: [batch, 1]

        return output
    


############################################
# PMAESによる実装
############################################
class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, h):
        w = torch.tanh(self.w(h))

        weight = self.v(w)
        weight = weight.squeeze(dim=-1)

        weight = torch.softmax(weight, dim=1)
        weight = weight.unsqueeze(dim=-1)
        out = torch.mul(h, weight.repeat(1, 1, h.size(2)))

        out = torch.sum(out, dim=1)

        return out


class EssayEncoder(nn.Module):
    def __init__(self, max_num, max_len, linguistic_feature_size, readability_feature_size, embed_dim=50, pos_vocab=None):
        super(EssayEncoder, self).__init__()
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