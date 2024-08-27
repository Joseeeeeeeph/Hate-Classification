import os

vicomtech_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/Vicomtech-hate-speech-dataset/all_files')
vicomtech_path_list = [os.path.join(vicomtech_path, f) for f in os.listdir(vicomtech_path)]
tweetdata_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/avaapm-hatespeech/tweetdata')
tweetdata_path_list = [os.path.join(tweetdata_path, f) for f in os.listdir(tweetdata_path)]

normalise = lambda x: x.lower().strip('!()}{[]\'"`,,.^-_+=/<>:;@#~|¬').replace('&', 'and').replace('colour', 'color').replace('centre', 'center').replace('favourite', 'favorite').replace('theatre', 'theater').replace(' ?', '?').replace('* * * *', '****').replace('* * *', '***').replace('* *', '**').replace('&', 'and').replace('colour', 'color').replace('centre', 'center').replace('favourite', 'favorite').replace('theatre', 'theater').replace(' ?', '?').replace('* * * *', '****').replace('* * *', '***').replace('* *', '**').replace('\n', ' ').replace('  ', ' ')

def normalise_files(files, directory):
    for file in files:
        with open(file, mode='r', encoding='utf-8') as fin:
            contents = fin.read()
            fin.close()
        with open(file, mode='w', encoding='utf-8') as fout:
            fout.write(normalise(contents))
            fout.close()

    print(f'Normalisation of source files complete for {directory}')

normalise_files(vicomtech_path_list, vicomtech_path)
normalise_files(tweetdata_path_list, tweetdata_path)