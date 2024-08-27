import model

normalise = lambda x: x.lower().strip('!()}{[]\'"`,.^-_+=/<>:;@#~|¬').replace('&', 'and').replace('fuck', '****').replace('shit', "****").replace('colour', 'color').replace('centre', 'center').replace('favourite', 'favorite').replace('theatre', 'theater').replace(' ?', '?').replace('* * * *', '****').replace('* * *', '***').replace('* *', '**')

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