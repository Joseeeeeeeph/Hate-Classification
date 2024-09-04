import os
import re

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+",
    flags=re.UNICODE
)

remove_links = lambda x: ' '.join([s for s in x.split() if 'http' not in s])
remove_punctuation = lambda x: ''.join([c for c in x if c not in '!?()}{[]\'"“”`,,.…^+=/<>:;@#~|¬'])
swap_strings = lambda x: x.replace('-', ' ').replace('_', ' ').replace('&amp', 'and').replace('&', 'and').replace('colour', 'color').replace('centre', 'center').replace('favourite', 'favorite').replace('theatre', 'theater').replace('* * * *', '****').replace('* * *', '***').replace('* *', '**').replace('\n', ' ').replace('  ', ' ')
remove_emojis = lambda x: emoji_pattern.sub(r'', x)

normalise = lambda x: swap_strings(remove_emojis(remove_punctuation(remove_links(x.lower()))))

def normalise_files(files, directory):
    for file in files:
        try:
            try:
                with open(file, mode='r') as fin:
                    contents = fin.read()
                    fin.close()
            except:
                with open(file, mode='r', encoding='utf-8') as fin:
                    contents = fin.read()
                    fin.close()
            try:
                with open(file, mode='w') as fout:
                    fout.write(normalise(contents))
                    fout.close()
            except:
                with open(file, mode='w', encoding='utf-8') as fout:
                    fout.write(normalise(contents))
                    fout.close()
        except:
            print(f'Could not normalise {file}')

    print(f'Normalisation of source files complete for {directory}')

vicomtech_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/Vicomtech-hate-speech-dataset/all_files')
vicomtech_path_list = [os.path.join(vicomtech_path, f) for f in os.listdir(vicomtech_path)]
tweetdata_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/avaapm-hatespeech/tweetdata')
tweetdata_path_list = [os.path.join(tweetdata_path, f) for f in os.listdir(tweetdata_path)]
ucberkeley_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/UCBerkeley-DLab-measuring-hate-speech/all_files')
ucberkeley_path_list = [os.path.join(ucberkeley_path, f) for f in os.listdir(ucberkeley_path)]
hatexplain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/hate-alert-HateXplain/all_files')
hatexplain_path_list = [os.path.join(hatexplain_path, f) for f in os.listdir(hatexplain_path)]

normalise_files(vicomtech_path_list, vicomtech_path)
normalise_files(tweetdata_path_list, tweetdata_path)
normalise_files(ucberkeley_path_list, ucberkeley_path)
normalise_files(hatexplain_path_list, hatexplain_path)