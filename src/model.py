import os
import torch
import pandas as pd
import pickle as pl
from time import time
from random import shuffle
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from multiprocessing import cpu_count

# Hyperparameters:
EPOCHS = 20
LR = 0.001
LR_DECAY = 0.1
BATCH_SIZE = 64
WEIGHT_DECAY = 0.0001
DROP_OUT = 0.5
PATIENCE = 3
LOSS_DELTA = 0.005
GRADIENT_CLIPPING = 0.1
EMBED_DIM = 256
K = 10

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

READ_FROM_CACHE = True

class HateSpeechDataset(Dataset):
    def __init__(self, contents):
        super(HateSpeechDataset, self).__init__()
        self.contents = contents
        self.nClasses = len(set(['hate', 'maybe hate', 'not hate']))

    def __getitem__(self, index):
        return self.contents[index]
    
    def __len__(self):
        return len(self.contents)
    
def get_class(annotations):
    score = 0
    for a in annotations:
        if a['label'] == 'hatespeech':
            score += 2
        elif a['label'] == 'offensive':
            score += 1
        elif a['label'] == 'normal':
            pass
        else:
            print('Unknown label:', a['label'])
            exit(1)

    return 2 if score >= 5 else 1 if score >= 2 else 0

def can_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False
    
def minutes(t):
        mins = int(t // 60)
        secs = t % 60
        if mins > 0:
            return '{:2d}m {:2.1f}s'.format(mins, secs)
        else:
            return ' {:2.1f}s'.format(secs)

def build(data, path, invert=False):
    data_entries = []
    dir = os.listdir(path)

    numericise_labels = lambda l: 2 if l == 'hate' else 1 if l == 'relation' else 0 if l == 'noHate' else l
    standardise_labels = lambda l: 2 if l >= 1.0 else 1 if l >= 0.5 else 0
    invert_labels = lambda l: 2 - l

    for entry in data: 
        id = str(entry[0])
        file = f'{id}.txt'
        if file not in dir:
            if can_int(entry[0]):
                id = f'{int(entry[0]):06d}'
                file = f'{id}.txt'
                if file not in dir:
                    continue
            else:
                continue

        if invert: entry[1] = invert_labels(entry[1])
        entry[1] = numericise_labels(entry[1])
        if type(entry[1]) == float: entry[1] = standardise_labels(entry[1])

        contents = open(os.path.join(path, file), mode='r', encoding='utf-8').read()
        entry.append(contents)

        if entry[1] == 'idk/skip':
            continue
        
        try:
            datum = tuple(entry[1:])
            if datum in data_entries:
                print('Duplicate entry:', datum)
                continue
            else:
                data_entries.append(datum)
        except:
            print('Could not allocate:', id)
    
    return data_entries

def cache(path, data):
    try:
        with open(path, 'wb') as file:
            pl.dump(data, file)
    except:
        print(f'Could not cache data at {path}')

vicomtech_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/Vicomtech-hate-speech-dataset')
vicomtech_all_path = os.path.join(vicomtech_path, 'all_files')
vicomtech_path_list = [os.path.join(vicomtech_all_path, f) for f in os.listdir(vicomtech_all_path)]

avaapm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/avaapm-hatespeech')
tweets_path = os.path.join(avaapm_path, 'tweetdata')
avaapm_path_list = [os.path.join(tweets_path, f) for f in os.listdir(tweets_path)]

ucberkeley_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/UCBerkeley-DLab-measuring-hate-speech')
ucberkeley_all_path = os.path.join(ucberkeley_path, 'all_files')
ucberkeley_path_list = [os.path.join(ucberkeley_all_path, f) for f in os.listdir(ucberkeley_all_path)]

hatexplain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/hate-alert-HateXplain')
hatexplain_all_path = os.path.join(hatexplain_path, 'all_files')
hatexplain_path_list = [os.path.join(hatexplain_all_path, f) for f in os.listdir(hatexplain_all_path)]

tdavidson_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/t-davidson-hate-speech-and-offensive-language')
tdavidson_all_path = os.path.join(tdavidson_path, 'all_files')
tdavidson_path_list = [os.path.join(tdavidson_all_path, f) for f in os.listdir(tdavidson_all_path)]

data_cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/data.pkl')

if os.path.isfile(data_cache_path) and READ_FROM_CACHE: 
    with open(data_cache_path, 'rb') as file:
        data = pl.load(file)

else:
    train_path = os.path.join(vicomtech_path, 'sampled_train')
    test_path = os.path.join(vicomtech_path, 'sampled_test')
    vicomtech_data = pd.read_csv(os.path.join(vicomtech_path, 'annotations_metadata.csv'), usecols=['file_id', 'label']).values.tolist()

    avaapm_id_list = [int(f[:-4]) for f in os.listdir(tweets_path)]
    avaapm_csv = pd.read_csv(os.path.join(avaapm_path, 'label.csv'), usecols=['TweetID', 'LangID', 'HateLabel'])
    avaapm_csv_en = avaapm_csv[avaapm_csv['LangID'] == 1]
    filtered_csv_en = avaapm_csv_en[avaapm_csv_en['TweetID'].isin(avaapm_id_list)]
    avaapm_data = filtered_csv_en[['TweetID', 'HateLabel']].values.tolist()

    ucberkeley_data = pd.read_csv(os.path.join(ucberkeley_path, 'label.csv'), usecols=['id', 'hate_speech_score']).values.tolist()

    hatexplain_df = pd.read_json(os.path.join(hatexplain_path, 'dataset.json')).T
    hatexplain_df = hatexplain_df[['post_id', 'annotators']]
    hatexplain_df['annotators'] = hatexplain_df['annotators'].apply(get_class)
    hatexplain_data = hatexplain_df.values.tolist()

    tdavidson_df = pd.read_csv(os.path.join(tdavidson_path, 'label.csv'), usecols=['class'])
    tdavidson_df['id'] = [f'{int(i):06d}' for i in tdavidson_df.index]
    tdavidson_data = tdavidson_df[['id', 'class']].values.tolist()

    all_data = []
    all_data += build(vicomtech_data, vicomtech_all_path)
    all_data += build(avaapm_data, tweets_path)
    all_data += build(ucberkeley_data, ucberkeley_all_path)
    all_data += build(hatexplain_data, hatexplain_all_path)
    all_data += build(tdavidson_data, tdavidson_all_path, invert=True)
    all_data = list(set(all_data))

    cache(data_cache_path, all_data)

dataset = HateSpeechDataset(all_data)

tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tokenizer.json')
if os.path.isfile(tokenizer_path) and READ_FROM_CACHE:
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path_list = vicomtech_path_list + avaapm_path_list + ucberkeley_path_list + hatexplain_path_list + tdavidson_path_list
    shuffle(tokenizer_path_list)
    tokenizer.train(tokenizer_path_list, trainer=BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]))
    tokenizer.save(tokenizer_path)

text_pipeline = lambda x: tokenizer.encode(x).ids

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

kf = KFold(n_splits=K, shuffle=True)

class ClassifierNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(ClassifierNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.dropout = nn.Dropout(DROP_OUT)
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.batch_norm(embedded)
        embedded = self.dropout(embedded)
        output = self.fc(embedded)
        return self.activation(output)

vocab_size = tokenizer.get_vocab_size()
total_accuracy = None
fold_results = []
model_path = os.path.join(os.path.dirname(os.path.realpath(os.path.dirname(__file__))), 'model.pth')

start_time = time()
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/{K}')
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, num_workers=cpu_count())
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, num_workers=cpu_count())
    
    model = ClassifierNN(vocab_size, EMBED_DIM, dataset.nClasses).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=LR_DECAY)
    scaler = GradScaler()

    best_loss = float('inf')
    patience_counter = 0
    total_accuracy = None

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500

        for id, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
            scaler.step(optimizer)
            scaler.update()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if id % log_interval == 0 and id > 0:
                print(f'epoch {epoch:3d} | {id:5d}/{len(dataloader):5d} batches | accuracy: {total_acc / total_count:8.3f}')
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

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time()
        train(train_loader)
        val_loss = 0
        validation_accuracy = evaluate(val_loader)
        val_loss /= len(val_loader)

        if total_accuracy is not None and total_accuracy > validation_accuracy:
            scheduler.step()
        else:
            total_accuracy = validation_accuracy
        print(f'end of epoch {epoch:3d} | time: {time() - epoch_start_time:5.2f}s | accuracy: {validation_accuracy:8.3f}')

        if val_loss < best_loss - LOSS_DELTA:
            torch.save(model.state_dict(), model_path)
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print('early stopping.')
            break

        fold_results.append((val_loss, validation_accuracy))

print("training finished in" + minutes(time() - start_time) + "\n\nChecking the results of test dataset.")
avg_loss = sum([result[0] for result in fold_results]) / K
avg_accuracy = sum([result[1] for result in fold_results]) / K
print(f'Average Validation Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()
    
model = model.to('cpu')