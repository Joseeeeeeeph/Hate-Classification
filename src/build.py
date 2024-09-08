import os
import requests
import datasets
import pandas as pd

def build_avaapm():
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/avaapm-hatespeech')
    storage_path = os.path.join(dataset_path, 'tweetdata')
    if not os.path.exists(storage_path): os.mkdir(storage_path)

    csv = pd.read_csv(os.path.join(dataset_path, 'label.csv'), usecols=['TweetID', 'LangID'])
    csv_en = csv[csv['LangID'] == 1]
    ids = csv_en['TweetID'].values.tolist()

    for id in ids:
        url = f"https://cdn.syndication.twimg.com/tweet-result?id={id}&token=a"

        try:
            text = requests.get(url).json()['text']
            file = open(os.path.join(storage_path, f'{id}.txt'), 'w')
            file.write(text)
            file.close()
            print(f"Getting tweet {id}: success")
        except:
            print(f"Getting tweet {id}: fail")

    print(f'\nTweet contents stored in {storage_path}')

def build_ucb():
    ucb_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/UCBerkeley-DLab-measuring-hate-speech')
    if not os.path.exists(ucb_path): os.mkdir(ucb_path)
    ucb_text_path = os.path.join(ucb_path, 'all_files')
    if not os.path.exists(ucb_text_path): os.mkdir(ucb_text_path)

    ucb_df = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')['train'].to_pandas()
    ucb_df['id'] = ['{:06d}'.format(int(i)) for i in ucb_df.index]
    ucb_df = ucb_df[['id', 'hate_speech_score', 'text']].drop_duplicates(subset='text')
    ucb_data = ucb_df[['id', 'text']].values.tolist()
    ucb_df.to_csv(os.path.join(ucb_path, 'label.csv'), index=False)

    previous = []

    for entry in ucb_data:
        contents = entry[1]
        if contents in previous: 
            continue
        else:
            file = open(os.path.join(ucb_text_path, entry[0] + '.txt'), 'w')
            file.write(contents)
            file.close()
            previous.append(contents)

    print(f'UCBerkeley-DLab-measuring-hate-speech contents stored in {ucb_text_path}')

def build_hatexplain():
    hatexplain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/hate-alert-HateXplain')
    hatexplain_text_path = os.path.join(hatexplain_path, 'all_files')
    if not os.path.exists(hatexplain_text_path): os.mkdir(hatexplain_text_path)
    hatexplain_df = pd.read_json(os.path.join(hatexplain_path, 'dataset.json')).T
    hatexplain_df = hatexplain_df[['post_id', 'post_tokens']]
    hatexplain_df['post_tokens'] = hatexplain_df['post_tokens'].apply(lambda s: ' '.join([c for c in s]))
    hatexplain_data = hatexplain_df.values.tolist()

    for entry in hatexplain_data:
        file = open(os.path.join(hatexplain_text_path, entry[0] + '.txt'), 'w')
        file.write(entry[1])
        file.close()

def remove_duplicates(path):
    previous = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            contents = f.read()
            f.close()
        if contents in previous:
            os.remove(os.path.join(path, file))
        else:
            previous.append(contents)

    print(f'Duplicates removed from {path}')

# >>>
build_avaapm()
build_ucb()
build_hatexplain()

remove_duplicates(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/Vicomtech-hate-speech-dataset/all_files'))