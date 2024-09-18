from torch import nn
from torch.utils.data import Dataset

# Hyperparameters:
EPOCHS = 32
LR = 0.00001
LR_DECREASE = 0.1
BATCH_SIZE = 96
WEIGHT_DECAY = 0.0001
DROP_OUT = 0.3
PATIENCE = 5
LOSS_DELTA = 0.001
GRADIENT_CLIPPING = 0.05
EMBED_DIM = 64
K = 10

# Neural Network:
class ClassifierNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, dropout):
        super(ClassifierNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.fc1 = nn.Linear(embed_dim, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, num_class)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size1)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size2)
        self.activation = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        hidden = self.fc1(embedded)
        hidden = self.batch_norm1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        hidden = self.fc2(hidden)
        hidden = self.batch_norm2(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        output = self.fc3(hidden)
        return output
    
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