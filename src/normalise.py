import os

all_files_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/hate-speech-dataset/all_files')
path_list = [os.path.join(all_files_path, f) for f in os.listdir(all_files_path)]

normalise = lambda x: x.lower().strip(' .').replace('\'\'', '"')

for file in path_list:
    with open(file, mode='r', encoding='utf-8') as fin:
        contents = fin.read()
        fin.close()
    with open(file, mode='w', encoding='utf-8') as fout:
        fout.write(normalise(contents))
        fout.close()

print('Normalisation of source files complete for {}'.format(all_files_path))