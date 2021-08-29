import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
import os
import json
import argparse

DATA_DIR = './'
MODEL_DIR = './model'

def save_weights(epoch, model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

def load_weights(epoch, model):
    model.load_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

def build_test_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(1,1)))
    for i in range(3):
        model.add(LSTM(256, return_sequences=(i != 2), stateful=True))
        model.add(Dropout(0.2))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    return model

def generate_music(epoch, num_char):
    with open(os.path.join(DATA_DIR, 'char_to_idx')) as f:
        char_to_idx = json.load(f)

    idx_to_char = dict(zip(char_to_idx.values(), char_to_idx.keys()))

    vocab_size = len(char_to_idx)
    batch_size = 16
    seq_len = 64
    model =  build_test_model(vocab_size)
    load_weights(epoch, model)

    sampled = []
    batch = np.zeros((1, 1))
    batch[0, 0] = np.random.randint(vocab_size)
    for i in range(num_char):
        if i ==0:
            result = model.predict_on_batch(batch).ravel()
            sample = np.random.choice(range(vocab_size), p=result)
            sampled.append(idx_to_char[sample])
        else:
            batch[0,0] = np.array([[sample]])
            result = model.predict_on_batch(batch).ravel()
            sample = np.random.choice(range(vocab_size), p=result)
            sampled.append(idx_to_char[sample])
    return ''.join(sampled)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample ABC notation generation from the trained model.')
    parser.add_argument('epoch', type=int, help='epoch checkpoint to sample from')
    parser.add_argument('--len', type=int, default=512, help='number of characters to sample (default 512)')
    args = parser.parse_args()

    print(generate_music(args.epoch, args.len))
