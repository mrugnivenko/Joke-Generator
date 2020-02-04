import numpy as np
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
import sys
sys.path.insert(1, '../')
import lib.constants as constants


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(id, destination):
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def get_empty_model(largest=constants.LARGEST, symbols=constants.SYMBOLS, lr=1e-4) -> Sequential:
    out = Sequential()
    out.add(LSTM(1024, return_sequences=True, input_shape=[largest, symbols]))
    out.add(Dropout(0.2))
    out.add(LSTM(1024, return_sequences=True))
    out.add(Dropout(0.2))
    out.add(LSTM(1024, return_sequences=True))
    out.add(Dropout(0.2))
    out.add(LSTM(1024, return_sequences=True))
    out.add(Dropout(0.2))
    out.add(TimeDistributed(Dense(symbols, activation='softmax')))
    adam_opti = Adam(lr=lr)
    out.compile(loss='categorical_crossentropy', optimizer=adam_opti, metrics=['categorical_accuracy'])
    return out


def load_weights_to_model(model, weights_id=None, optimizer=Adam(lr=0.0001), path=None):
    """
    functions that load weights to model
    :param model: here we gonna load weights
    :param weights_id: I suppose that we are to store weights in google
    :param path: or in directory
    :param optimizer:
    :return:
    """
    if path is None and weights_id is None:
        print('Lol, no idea where are weights. Give me id or path, lol')
        return 1
    if path is None:
        weight_path = 'weights.hdf5'
        download_file_from_google_drive(weights_id, weight_path)
    else:
        weight_path = path

    model.load_weights(weight_path)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['categorical_accuracy'])


def get_jokes(model, init_word='мужик', jokes_num=1, joke_len=30, largest=constants.LARGEST, symbols=constants.SYMBOLS,
              decoder=constants.num_char) -> list:
    out = []
    for jokes in range(jokes_num):
        gen = np.zeros([1, largest, symbols], dtype='int8')
        gen[0][0][1] = 1
        joke = init_word
        for i in range(1, joke_len):
            pred = model.predict(gen)
            pred = pred[0][i - 1]
            letter = np.random.choice(symbols, p=pred)
            if letter == 0 or letter == 1:
                break
            joke += decoder[letter]
            gen[0][i][letter] = 1
        out.append(joke)
    return out
