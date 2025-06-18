import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# Attention LSTM architecture:
class AttentionLSTMClassifier(nn.Module):
    def __init__(self, nClasses, tokenizer, dropout=0.3, embedding_dim=256, hidden_dim=128, num_layers=2, attention_dim=32):
        super().__init__()

        self.embedding = nn.Embedding(
            len(tokenizer.get_vocab()), 
            embedding_dim, 
            padding_idx=tokenizer.token_to_id('[PAD]')
        )

        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout
        )

        self.attn_w = nn.Linear(2 * hidden_dim, attention_dim, bias=True)
        self.attn_v = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, nClasses)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, _ = self.lstm(embedded)

        u = torch.tanh(self.attn_w(output))
        scores = self.attn_v(u).squeeze(-1) 
        alpha = F.softmax(scores, dim=1)

        context = torch.bmm(alpha.unsqueeze(1), output)
        context = context.squeeze(1) 

        dropped = self.dropout(context)
        logits = self.classifier(dropped)

        return logits

# Dataset Structure:
class HateSpeechDataset(Dataset):
    def __init__(self, contents):
        super(HateSpeechDataset, self).__init__()
        self.contents = contents
        self.nClasses = len({'hate', 'maybe hate', 'not hate'})

    def __getitem__(self, index):
        return self.contents[index]
    
    def __len__(self):
        return len(self.contents)