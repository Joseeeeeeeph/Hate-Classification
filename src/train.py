import os
import pickle as pl
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch import GradScaler
from torch import autocast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from time import time
from sklearn.model_selection import StratifiedKFold

from architecture import AttentionLSTMClassifier, HateSpeechDataset

K = 10
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-5
LR_DECREASE = 0.5
MINIMUM_LR = 1e-6
GRADIENT_CLIPPING = 1.0
EPOCHS = 30
PATIENCE = 2
LOSS_DELTA = 1e-4

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.flash_attention = True
    scaler = GradScaler()
else:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.backends.cuda.flash_attention = False
    scaler = None
    torch.set_num_threads(os.cpu_count())

READ_FROM_CACHE = True
    
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
            return '{:3d}m {:2.1f}s'.format(mins, secs)
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
        all_data = pl.load(file)[0]

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
    np.random.shuffle(tokenizer_path_list)
    tokenizer.train(tokenizer_path_list, trainer=BpeTrainer(special_tokens=['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']))
    tokenizer.save(tokenizer_path)

text_pipeline = lambda x: tokenizer.encode(x).ids

all_tokenised = []
for label, text in all_data:
    ids = text_pipeline(text)
    all_tokenised.append((label, ids))
dataset = HateSpeechDataset(all_tokenised)

def collate_batch(batch):
    labels, seqs = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.int64)
    lengths = [len(s) for s in seqs]
    seq_tensors = [torch.tensor(s, dtype=torch.int64) for s in seqs]
    pad_id = tokenizer.token_to_id('[PAD]') if hasattr(tokenizer, 'token_to_id') else 0
    padded = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_id)
    return labels.to(device), padded.to(device), torch.tensor(lengths).to(device)

kf = StratifiedKFold(n_splits=K, shuffle=True)

vocab_size = tokenizer.get_vocab_size()
total_accuracy = None
fold_results = []
all_false_positives = []
all_false_negatives = []
total_val_size = 0
overall_best_loss = float('inf')

model_path = os.path.join(os.path.dirname(os.path.realpath(os.path.dirname(__file__))), 'model.pth')

start_time = time()
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, [d[0] for d in all_data])):
    print(f'Fold {fold + 1}/{K}')
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    model = AttentionLSTMClassifier(dataset.nClasses, tokenizer).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LR_DECREASE, patience=2, min_lr=MINIMUM_LR)

    best_loss = float('inf')
    patience_counter = 0
    total_accuracy = None
    display_training = False

    false_positives_hate = 0
    false_negatives_hate = 0
    false_positives_maybe = 0
    false_negatives_maybe = 0

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500

        global display_training, train_loss

        for id, (label, text, seq_lengths) in enumerate(dataloader):
            optimizer.zero_grad()
            print(f'Batch {id + 1}/{len(dataloader)} loaded', end='\r')
            with autocast(device_type=device.type):
                predicted_label = model(text, seq_lengths)
                loss = criterion(predicted_label, label)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                optimizer.step()
            
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            train_loss += loss.item()
            if id % log_interval == 0 and id > 0:
                display_training = True
                score = total_acc / total_count
                print(f'epoch {epoch:3d} | {id:5d}/{len(dataloader):5d} batches | accuracy: {score:8.3f}')
                total_acc, total_count = 0, 0

    def evaluate(dataloader):
        global val_loss, false_positives_hate, false_negatives_hate, false_positives_maybe, false_negatives_maybe
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad(), autocast(device_type=device.type):
            for _, (label, text, seq_lengths) in enumerate(dataloader):
                predicted_label = model(text, seq_lengths)
                total_acc += (predicted_label.argmax(1) == label).sum().item()

                false_positives_hate += ((predicted_label.argmax(1) == 2) & (label != 2)).sum().item()
                false_negatives_hate += ((predicted_label.argmax(1) != 2) & (label == 2)).sum().item()
                false_positives_maybe += ((predicted_label.argmax(1) == 1) & (label != 1)).sum().item()
                false_negatives_maybe += ((predicted_label.argmax(1) != 1) & (label == 1)).sum().item()

                total_count += label.size(0)
                val_loss += criterion(predicted_label, label).item()
        return total_acc / total_count

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time()

        train_loss = 0
        train(train_loader)
        train_loss /= len(train_loader)

        val_loss = 0
        validation_accuracy = evaluate(val_loader)
        val_loss /= len(val_loader)

        if total_accuracy is not None and total_accuracy > validation_accuracy:
            scheduler.step(metrics=validation_accuracy)
        else:
            total_accuracy = validation_accuracy

        epoch_length = time() - epoch_start_time
        print(f'end of epoch {epoch:3d} | time: {epoch_length:5.2f}s | accuracy: {validation_accuracy:8.3f} | training loss: {train_loss:8.3f} | validation loss: {val_loss:8.3f}')

        if val_loss < best_loss - LOSS_DELTA:
            best_loss = val_loss
            patience_counter = 0
            if val_loss < overall_best_loss:
                overall_best_loss = val_loss
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print('early stopping.')
            break

    fold_results.append((val_loss, validation_accuracy))
    total_val_size += len(val_subset)

print('training finished in' + minutes(time() - start_time) + '\n\nChecking the results:')

avg_loss = sum([result[0] for result in fold_results]) / K
avg_accuracy = sum([result[1] for result in fold_results]) / K
avg_false_positive_hate = sum([instance[0] for instance in all_false_positives]) / total_val_size / K
avg_false_negative_hate = sum([instance[0] for instance in all_false_negatives]) / total_val_size / K
avg_false_positive_maybe = sum([instance[1] for instance in all_false_positives]) / total_val_size / K
avg_false_negative_maybe = sum([instance[1] for instance in all_false_negatives]) / total_val_size / K

print(f'Average Validation Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}\nFalse Positive Rate (Hate): {avg_false_positive_hate:.4f}, False Negative Rate (Hate): {avg_false_negative_hate:.4f}, Average False Positives (Maybe Hate): {avg_false_positive_maybe:.4f}, Average False Negatives (Maybe Hate): {avg_false_negative_maybe:.4f}')