import os
import requests
import pandas as pd

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/avaapm-hatespeech')
storage_path = os.path.join(dataset_path, 'tweetdata')

csv = pd.read_csv(os.path.join(dataset_path, 'label.csv'), usecols=['TweetID', 'LangID', 'HateLabel'])
csv_en = csv[csv['LangID'] == 1]
ids = csv_en['TweetID'].values.tolist()

for id in ids:
    url = f"https://cdn.syndication.twimg.com/tweet-result?id={id}&token=a"

    try:
        text = requests.get(url).json()['text']
        file = open(os.path.join(storage_path, f'{id}.txt'), 'w')
        file.write(text)
        print(f"Getting tweet {id}: success")
    except:
        print(f"Getting tweet {id}: fail")

print(f'\nTweet contents stored in {storage_path}')