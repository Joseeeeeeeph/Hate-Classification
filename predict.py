import os
import torch
from tokenizers import Tokenizer

from src.architecture import AttentionLSTMClassifier, HateSpeechDataset

model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pth')
tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src/tokenizer.json')

try:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    text_pipeline = lambda x: tokenizer.encode(x).ids

    model = AttentionLSTMClassifier(HateSpeechDataset([]).nClasses, tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
except:
    print('Training Model:')
    from train import model, text_pipeline

def predict(text):
    with torch.no_grad():
        tokenised_text = text_pipeline(text)
        text_tensor = torch.tensor([tokenised_text], dtype=torch.int64)
        length = torch.tensor([len(tokenised_text)], dtype=torch.int64)
        if len(text) <= 1:
            return 0
        else:
            output = model(text_tensor, length)
            return output.argmax(1).item()
    
model.eval()

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
            print(f'"{message}" is {classify(predict(normalise(message)))}\n')

if __name__ == '__main__':
    main()