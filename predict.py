import os
import torch
from tokenizers import Tokenizer

from src.architecture import ClassifierNN, HateSpeechDataset, EMBED_DIM, DROP_OUT

model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pth')
tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src/tokenizer.json')

try:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    text_pipeline = lambda x: tokenizer.encode(x).ids

    model = ClassifierNN(tokenizer.get_vocab_size(), EMBED_DIM, HateSpeechDataset([]).nClasses, DROP_OUT)
    model.load_state_dict(torch.load(model_path, weights_only=False))
except:
    print('Training Model:')
    from src.model import model, text_pipeline

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()
    
model = model.to('cpu').eval()

remove_links = lambda x: ' '.join([s for s in x.split() if 'http' not in s])
remove_punctuation = lambda x: ''.join([c for c in x if c not in r'!?()}{[]\'"“”`,,.…^+=/:;@#~|%¬\\£$€¥¢'])
swap_strings = lambda x: x.replace('-', ' ').replace('_', ' ').replace('&amp', 'and').replace('&', 'and').replace('colour', 'color').replace('centre', 'center').replace('favourite', 'favorite').replace('theatre', 'theater').replace('* * * *', '****').replace('* * *', '***').replace('* *', '**').replace('\n', ' ').replace('<user>', '').replace('<url>', '').replace('<censored>', '****').replace('<', '').replace('>', '').replace('  ', ' ')
normalise = lambda x: swap_strings(remove_punctuation(remove_links(x.lower())))

classify = lambda x: 'hate' if x == 2 else 'maybe hate' if x == 1 else 'not hate' if x == 0 else 'unclear/classification error'

def main():
    print('Classifying inputs. Enter "!quit" to exit.')
    while True:
        message = input('Enter text to classify: ')

        if message == '!quit':
            exit()
        else:
            print('"' + message + '" is {}\n'.format(classify(predict(normalise(message), text_pipeline))))

if __name__ == "__main__":
    main()