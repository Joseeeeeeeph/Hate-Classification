import torch

from model import ClassifierNN, training_dataloader, validation_dataloader, testing_dataloader, device, vocab_size, emsize

# Hyper-hyperparameters: -------------------
start = 1
end = 3
step = 0.1
# ------------------------------------------

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
lr = ptr('LR', 5)
epochs = ptr('Epochs', 10)
weight_decay = ptr('Weight Decay', 0)
dropout = ptr('Dropout', 0.3)
patience = ptr('Patience', 5)

best_accuracy = 0
best_val = None
classes = 2
scale = 1
val_loss = 0

def hyperparameter_tuning(hyperparameter, controls):
    global best_accuracy, best_val

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

    print('\nTuning complete with ' + hyperparameter.__str__() + ' = {} with an accuracy of {}'.format(best_val, best_accuracy))

# >>>
hyperparameter_tuning(lr, (start, end, step))