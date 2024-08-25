import os
import torch
import pandas as pd
from time import time
from math import ceil
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
#from tokenizers.pre_tokenizers import Whitespace

# Hyperparameters:
EPOCHS = 10
LR = 2
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0001
DROP_OUT = 0.3
PATIENCE = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class HateSpeechDataset(Dataset):
    def __init__(self, contents):
        super(HateSpeechDataset, self).__init__()
        self.contents = contents
        self.nClasses = len(set(['hate', 'noHate']))

    def __getitem__(self, index):
        return self.contents[index]
    
    def __len__(self):
        return len(self.contents)
    
def holdout(data_list, all_path, train_path, test_path):
    head = len(all_path) + 1
    tail = -len('.txt')
    training = [f[:tail] for f in os.listdir(train_path)]
    testing = [f[:tail] for f in os.listdir(test_path)]
    remaining = [f[head:tail] for f in data_list if f[head:tail] not in training and f[head:tail] not in testing]
    partition = int(ceil(0.7 * len(remaining)))
    training = training + remaining[:partition]
    testing = testing + remaining[partition:]
    
    return training, testing

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/hate-speech-dataset')
all_files_path = os.path.join(dataset_path, 'all_files')
train_path = os.path.join(dataset_path, 'sampled_train')
test_path = os.path.join(dataset_path, 'sampled_test')
path_list = [os.path.join(all_files_path, f) for f in os.listdir(all_files_path)]
train_list, test_list = holdout(path_list, all_files_path, train_path, test_path)

train_data = []
test_data = []
data = pd.read_csv(os.path.join(dataset_path, 'annotations_metadata.csv'), usecols=['file_id', 'label']).values.tolist()
for entry in data:
    id = entry[0]
    contents = open(os.path.join(all_files_path, f'{id}.txt'), 'r').read()
    entry.append(contents)
    if entry[1] == 'idk/skip':
        continue
    elif id in train_list:
        train_data.append(tuple(entry[1:]))
    elif id in test_list:
        test_data.append(tuple(entry[1:]))
    else:
        print('Could not allocate:', id)

training_dataset = HateSpeechDataset(train_data)
testing_dataset = HateSpeechDataset(test_data)
n = int(len(training_dataset) * 0.95)
train_split, valid_split = random_split(training_dataset, [n, len(training_dataset) - n])

tokenizer = Tokenizer(BPE())
#tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(path_list, trainer=BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]))
text_pipeline = lambda x: tokenizer.encode(x).ids

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(1 if _label == 'hate' or _label == 'relation' else 0)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

training_dataloader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
validation_dataloader = DataLoader(valid_split, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
testing_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

class ClassifierNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(ClassifierNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.dropout = nn.Dropout(DROP_OUT)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

vocab_size = tokenizer.get_vocab_size()
emsize = 64
total_accuracy = None

model = ClassifierNN(vocab_size, emsize, training_dataset.nClasses).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500

    for id, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if id % log_interval == 0 and id > 0:
            print('epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(epoch, id, len(dataloader), total_acc / total_count))
            total_acc, total_count = 0, 0

def evaluate(dataloader):
    global val_loss
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            val_loss += criterion(predicted_label, label).item()
    return total_acc / total_count

best_loss = float('inf')
patience_counter = 0
start_time = time()

print('\ntraining model:')
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time()
    train(training_dataloader)
    val_loss = 0
    validation_accuracy = evaluate(validation_dataloader)
    val_loss /= len(validation_dataloader)

    if total_accuracy is not None and total_accuracy > validation_accuracy:
        scheduler.step()
    else:
        total_accuracy = validation_accuracy
    print('end of epoch {:3d} | time: {:5.2f}s | accuracy {:8.3f}'.format(epoch, time() - epoch_start_time, validation_accuracy))

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        print('early stopping.')
        break


def minutes(t):
    mins = int(t // 60)
    secs = t % 60
    if mins > 0:
        return '{:2d}m {:2.1f}s'.format(mins, secs)
    else:
        return ' {:2.1f}s'.format(secs)

print("training finished in" + minutes(time() - start_time) + "\n\nChecking the results of test dataset.")
test_accuracy = evaluate(testing_dataloader)
print("test accuracy {:8.3f}\n".format(test_accuracy))

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()
    
model = model.to('cpu')