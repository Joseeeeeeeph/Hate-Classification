import model

remove_links = lambda x: ' '.join([s for s in x.split() if 'http' not in s])
remove_punctuation = lambda x: ''.join([c for c in x if c not in '!()}{[]\'"“”`,,.…^+=/<>:;@#~|¬'])
swap_strings = lambda x: x.replace('-', ' ').replace('_', ' ').replace('&amp', 'and').replace('&', 'and').replace('colour', 'color').replace('centre', 'center').replace('favourite', 'favorite').replace('theatre', 'theater').replace(' ?', '?').replace('* * * *', '****').replace('* * *', '***').replace('* *', '**').replace('\n', ' ').replace('  ', ' ')
normalise = lambda x: swap_strings(remove_punctuation(remove_links(x.lower())))

classify = lambda x: 'hate' if x == 2 else 'maybe hate' if x == 1 else 'not hate' if x == 0 else 'unclear/classification error'

def main():
    print('Classifying inputs. Enter "!quit" to exit.')
    while True:
        message = input('Enter text to classify: ')

        if message == '!quit':
            exit()
        else:
            print('"' + message + '" is {}\n'.format(classify(model.predict(normalise(message), model.text_pipeline))))

if __name__ == "__main__":
    main()