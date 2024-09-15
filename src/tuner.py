import os
import torch
import itertools
import pickle as pl
from numpy import linspace
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from model import ClassifierNN, device, vocab_size, collate_batch, dataset, kf

class ptr():
    def __init__(self, id, value): 
        self.value = value
        self.id = id

    def get(self): return self.value
    def set(self, value): self.value = value
    def __str__(self): return str(self.id)

class ModifiedNN(ClassifierNN):
    def __init__(self, vocab, embed_dim, nClasses, dropout):
        super(ModifiedNN, self).__init__(vocab, embed_dim, nClasses)
        self.dropout = torch.nn.Dropout(dropout)

# Hyperparameters:
epochs = ptr('Epochs', 32)
lr = ptr('LR', 0.001)
lr_decay = ptr('LR Decay', 0.1)
batch_size = ptr('Batch Size', 64)
weight_decay = ptr('Weight Decay', 0.0001)
dropout = ptr('Dropout', 0.5)
patience = ptr('Patience', 3)
loss_delta = ptr('Loss Delta', 0.02)
gradient_clipping = ptr('Gradient Clipping', 0.1)
embed_dim = ptr('Embedding Dimension', 173)
k = ptr('k', 10)

VAL_PER_GV = 3

def store(data):
    pkl_name = 'hyperparameters.pkl'
    iteration = 0
    cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '__pycache__')
    if not os.path.exists(cache_path): os.mkdir(cache_path)
    file_path = os.path.join(cache_path, pkl_name)

    while True:
        if os.path.exists(file_path):
            iteration += 1
            pkl_name = 'hyperparameters' + f'{iteration}.pkl'
            file_path = os.path.join(cache_path, pkl_name)
        else:
            break
    
    with open(file_path, 'wb') as file: pl.dump(data, file)

def load_states():
    cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '__pycache__')
    if not os.path.exists(cache_path): os.mkdir(cache_path)
    file_path = os.path.join(cache_path, 'prevstates.pkl')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file: return pl.load(file)[0]
    else:
        return []
    
def cache_states(states):
    cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '__pycache__')
    file_path = os.path.join(cache_path, 'prevstates.pkl')
    with open(file_path, 'wb') as file: pl.dump(states, file)

best_avg_loss = float('inf')

def hyperparameter_tuning(hyperparameters, generators, write=False):
    value_gen = [linspace(g[0], g[1], VAL_PER_GV).tolist() for g in generators]
    indices = list(range(VAL_PER_GV))

    best_hyperparameters = {hyperparameter.__str__(): None for hyperparameter in hyperparameters}
    hyperparameter_value_gens = dict(zip([h.__str__() for h in hyperparameters], value_gen))
    for h in hyperparameters: h.set(hyperparameter_value_gens[h.__str__()][0])

    previous_states = load_states()

    i = -1
    for state in itertools.product(hyperparameters, indices):
        if state in previous_states:
            continue
        else:
            previous_states.append(state)
            cache_states(previous_states)

        h = state[0]
        idx = state[1]
        h.set(hyperparameter_value_gens[h.__str__()][idx])
        i += 1

        fold_results = []
        total_accuracy = None

        for _, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size.get(), shuffle=True, collate_fn=collate_batch)
            val_loader = DataLoader(val_subset, batch_size=batch_size.get(), shuffle=True, collate_fn=collate_batch)
            
            model = ModifiedNN(vocab_size, embed_dim.get(), dataset.nClasses, dropout.get()).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr.get(), weight_decay=weight_decay.get())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=lr_decay.get())
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping.get())
                    scaler.step(optimizer)
                    scaler.update()
                    total_acc += (predicted_label.argmax(1) == label).sum().item()
                    total_count += label.size(0)
                    if id % log_interval == 0 and id > 0:
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

            for _ in range(1, epochs.get() + 1):
                train(train_loader)
                val_loss = 0
                validation_accuracy = evaluate(val_loader)
                val_loss /= len(val_loader)

                if total_accuracy is not None and total_accuracy > validation_accuracy:
                    scheduler.step()
                else:
                    total_accuracy = validation_accuracy

                if val_loss < best_loss - loss_delta.get():
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience.get():
                    break

            fold_results.append((val_loss, validation_accuracy))

        avg_loss = sum([result[0] for result in fold_results]) / k.get()
        avg_accuracy = sum([result[1] for result in fold_results]) / k.get()
        print(f'Average Validation Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_hyperparameters = {hyperparameter.__str(): hyperparameter.get() for hyperparameter in hyperparameters}
            if write: store(best_hyperparameters)

# >>>
hyperparameters = [epochs, lr, lr_decay, batch_size, weight_decay, dropout, patience, loss_delta, gradient_clipping, embed_dim, k]
generators = [
    (30, 30),               # Epochs
    (0.0001, 0.002),        # LR
    (0.1, 0.5),             # LR decay
    (32, 256),              # Batch Size
    (0.00001, 0.0001),      # Weight Decay
    (0.0, 0.5),             # Dropout
    (3, 6),                 # Patience
    (0, 0.5),               # Loss Delta
    (0.1, 0.2),             # Gradient Clipping
    (64, 512),              # Embed Dim
    (5, 15)                 # k
]

hyperparameter_tuning(hyperparameters, generators, write=True)