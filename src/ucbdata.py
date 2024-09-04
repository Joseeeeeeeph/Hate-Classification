import os
import datasets

ucb_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/UCBerkeley-DLab-measuring-hate-speech')
if not os.path.exists(ucb_path): os.mkdir(ucb_path)
ucb_text_path = os.path.join(ucb_path, 'all_files')
if not os.path.exists(ucb_text_path): os.mkdir(ucb_text_path)

ucb_df = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')['train'].to_pandas()
ucb_df['id'] = ['{:06d}'.format(int(i)) for i in ucb_df.index]
ucb_data = ucb_df[['id', 'text']].values.tolist()
ucb_df[['id', 'hate_speech_score', 'text']].to_csv(os.path.join(ucb_path, 'label.csv'), index=False)

for entry in ucb_data:
    file = open(os.path.join(ucb_text_path, entry[0] + '.txt'), 'w')
    file.write(entry[1])
    file.close()

print(f'UCBerkeley-DLab-measuring-hate-speech contents stored in {ucb_text_path}')