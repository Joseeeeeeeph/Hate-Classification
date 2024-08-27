import os
import torch
import pickle as pl

from model import ClassifierNN, training_dataloader, validation_dataloader, testing_dataloader, device, vocab_size, emsize

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
lr = ptr('LR', 2)
epochs = ptr('Epochs', 16)
weight_decay = ptr('Weight Decay', 0.0001)
dropout = ptr('Dropout', 0.3)
patience = ptr('Patience', 4)

def reset():
    lr.set(2)
    epochs.set(16)
    weight_decay.set(0.0001)
    dropout.set(0.3)
    patience.set(4)

def store(hyperparameter, values):
    pkl_name = hyperparameter + '.pkl'
    iteration = 0
    cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '__pycache__')
    if os.path.isfile(cache_path):
        os.makedirs(cache_path)
    file_path = os.path.join(cache_path, pkl_name)

    while True:
        if os.path.isfile(file_path):
            iteration += 1
            pkl_name = hyperparameter + f'{iteration}.pkl'
            file_path = os.path.join(cache_path, pkl_name)
        else:
            break
    
    with open(file_path, 'wb') as file: pl.dump(values, file)

best_accuracy = 0
best_val = None
best_value_list = []
classes = 2
scale = 1
val_loss = 0

def hyperparameter_tuning(hyperparameter, controls, write=False):
#    controls := (start, end, step) where:
#       start := starting value of hyperparameter
#       end := ending value of hyperparameter
#       step := how many values inbetween
#    write := whether to store the results in a .pkl file

    global best_accuracy, best_val
    all_scores = []

    start, end, step = controls
    if type(start) is not int or type(end) is not int or type(step) is not int:
        scale = step
        start = int(start / scale)
        end = int(end / scale)
        step = int(step / scale)

    digits = lambda x: len(str(x - round(x)))
    display_digits = max(digits(start), digits(end), digits(step))

    for i in range(start, end, step):
        hyperparameter.set(round(float(i * scale), display_digits))

        total_accuracy = None
        model = ModifiedNN(vocab_size, emsize, classes, dropout.get()).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr.get(), weight_decay=weight_decay.get())
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

        for _ in range(1, int(epochs.get()) + 1):
            train(training_dataloader)

            val_loss = 0
            validation_accuracy = evaluate(validation_dataloader)
            val_loss /= len(validation_dataloader)

            if total_accuracy is not None and total_accuracy > validation_accuracy:
                scheduler.step()
            else:
                total_accuracy = validation_accuracy

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= int(patience.get()):
                break

        test_accuracy = evaluate(testing_dataloader)
        print('accuracy of {:2.3f} for '.format(test_accuracy) + hyperparameter.__str__() + ' = {}'.format(hyperparameter.get()))
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_val = hyperparameter.get()

    best_value_list.append((hyperparameter.__str__(), best_val))
    if write: store(hyperparameter.__str__(), all_scores)
    print('\nTuning complete with ' + hyperparameter.__str__() + ' = {} with an accuracy of {}'.format(best_val, best_accuracy))
    reset()

# >>>
hyperparameter_tuning(lr, (0.1, 6, 0.1), write=True)
hyperparameter_tuning(weight_decay, (0, 0.01, 0.0001), write=True)
hyperparameter_tuning(weight_decay, (0, 0.1, 0.001), write=True)
hyperparameter_tuning(weight_decay, (0, 1, 0.1), write=True)
hyperparameter_tuning(dropout, (0, 0.8, 0.1), write=True)
hyperparameter_tuning(patience, (2, 10, 1), write=True)

print(f'\nBest hyperparameters:\n {best_value_list}')