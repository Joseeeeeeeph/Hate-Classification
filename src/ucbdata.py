import os
import datasets

ucb_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/UCBerkeley-DLab-measuring-hate-speech')
if not os.path.exists(ucb_path): os.mkdir(ucb_path)

ucb_df = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')['train'].to_pandas()
ucb_data = ucb_df[['comment_id', 'text']].values.tolist()

for entry in ucb_data:
    id = entry[0]
    file = open(os.path.join(ucb_path, f'{id}.txt'), 'w')
    file.write(entry[1])
    file.close()

print(f'\nUCBerkeley-DLab-measuring-hate-speech contents stored in {ucb_path}')