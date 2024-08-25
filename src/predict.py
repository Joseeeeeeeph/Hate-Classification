import model

while True:
    message = input('Enter text to classify: ')
    normalise = lambda x: x.lower().strip('!()}{[]\'"`,.^-_+=/<>:;@#~|¬').replace('&', 'and').replace('fuck', '* * * *').replace('shit', "* * * *").replace('?', ' ? ').replace('colour', 'color')
    if message == '!quit':
        exit()
    else:
        print('"' + message + '" is {}\n'.format('hate' if model.predict(normalise(message), model.text_pipeline) == 1 else 'not hate'))