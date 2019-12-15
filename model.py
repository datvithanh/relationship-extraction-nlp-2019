#model.py
import torch
import torch.nn as nn 
import torch.nn.functional as F

class Let(nn.Module):
    def __init__(self, latent_size, num_type = 3):
        # num_type = scalar(int)
        # latent_size = hidden = scalar(int)
        super(Let, self).__init__()
        self.let_vector = nn.Parameter(torch.randn(num_type, latent_size))

    def forward(self, h_e1, h_e2):
        # e1, e2 = (batch, hidden)
        a1 = F.softmax(torch.mm(h_e1, torch.t(self.let_vector)), dim=-1)
        a2 = F.softmax(torch.mm(h_e2, torch.t(self.let_vector)), dim=-1)
        t1 = torch.mm(a1, self.let_vector)
        t2 = torch.mm(a2, self.let_vector)
        return t1, t2

class Attention(nn.Module):
    def __init__(self, hidden_size, pos_embedding_dim, attention_size, pos_vocab_size):
        # attention_size = scalar(int)
        super(Attention, self).__init__()
        self.let = Let(hidden_size)
        self.denseE = nn.Linear(hidden_size*4, attention_size, bias=False)
        self.denseP = nn.Linear(hidden_size + pos_embedding_dim*2, attention_size, bias=False)
        self.u = nn.Parameter(torch.randn(attention_size, ))

    def forward(self, inputs, e1, e2, p1, p2):
        # inputs = (batch, seq_len, hidden)
        # e1, e2 = (batch, )
        # p1, p2 = (batch, seq_len, dist_emb_size)

        # can pytorch track slicing with autograd? Yasss
        h_e1 = inputs.gather(1, e1.view(-1, 1, 1).expand(inputs.size(0), 1, inputs.size(2)))[:, 0, :]
        h_e2 = inputs.gather(1, e2.view(-1, 1, 1).expand(inputs.size(0), 1, inputs.size(2)))[:, 0, :]

        t1, t2 = self.let(h_e1, h_e2)

        h_e = torch.cat([h_e1, t1, h_e2, t2], dim = -1) # (batch, hidden*4)
        h_p = torch.cat([inputs, p1, p2], dim=-1) # (batch, seq_len, hidden + dist_emb_size*2)

        h_e = self.denseE(h_e)[:, None, :] # (batch, attn)
        h_p = self.denseP(h_p) # (batch, seq_len, attn)

        alpha = F.softmax(h_p.add(h_e).matmul(self.u), dim=-1) # (batch, seq_len)

        output = torch.bmm(alpha[:, None, :], inputs)[:, 0, :] # (batch, hidden)

        return output

class Attention_bilstm_let(nn.Module):
    def __init__(self, embedding_dim, hidden_size, pos_embedding_dim, attention_size, num_classes, pos_vocab_size=162):
        super(Attention_bilstm_let, self).__init__()
        
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim)

        self.slf_attn = nn.MultiheadAttention(embedding_dim, 4)
        
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, dropout=0.3)

        self.attn = Attention(hidden_size*2, pos_embedding_dim, attention_size, pos_vocab_size)

        self.attn_dropout = nn.Dropout(0.5)

        self.ln = nn.Linear(hidden_size*2, num_classes)

    def forward(self, X, e1, e2, p1, p2):
        # X = (batch, seq_len, )
        X, _ = self.slf_attn(X, X, X) 
        X, _ = self.bilstm(X)
        # X = torch.transpose(X, 0, 1)

        p1 = self.pos_embedding(p1)
        p2 = self.pos_embedding(p2)

        attn = self.attn(X, e1, e2, p1, p2)

        attn_drop = self.attn_dropout(attn)

        logits = self.ln(attn_drop)
        
        return logits