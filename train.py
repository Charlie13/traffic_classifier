import sys
import os
import json
import pandas
import numpy
import optparse
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
from keras.models import load_model

batch_size = 32
num_epochs = 3

"""
Step 1 : Load and preprocess data
"""
def preprocess(csv_file):
    dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
    dataset = dataframe.sample(frac=1).values

    X = dataset[:,0]
    Y = dataset[:,1]

    for index, item in enumerate(X):
        # Quick hack to space out json elements
        reqJson = json.loads(item, object_pairs_hook=OrderedDict)
        del reqJson['timestamp']
        del reqJson['headers']
        del reqJson['source']
        del reqJson['route']
        del reqJson['responsePayload']
        X[index] = json.dumps(reqJson, separators=(',', ':'))

    # Vectorize a text corpus, by turning each text into either a sequence of integers
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X)

    """ Extract and save word dictionary """
    word_dict_file = 'build/word-dictionary.json'
    # Create file if not exist
    if not os.path.exists(os.path.dirname(word_dict_file)):
        os.makedirs(os.path.dirname(word_dict_file))
    # Save content
    with open(word_dict_file, 'w') as outfile:
        json.dump(tokenizer.word_index, outfile, ensure_ascii=False)


    num_words = len(tokenizer.word_index)+1
    # Transform all text to a sequence of integers
    X = tokenizer.texts_to_sequences(X)

    max_log_length = 1024
    train_size = int(len(dataset) * .75)

    X_processed = sequence.pad_sequences(X, maxlen=max_log_length)
    # Divide to train dataset
    X_train = X_processed[0:train_size]
    Y_train = Y[0:train_size]
    # and test dataset
    X_test = X_processed[train_size:len(X_processed)]
    Y_test = Y[train_size:len(Y)]

    return X_train, Y_train, X_test, Y_test, num_words, max_log_length


"""
Step 2: Build and train model
"""
def train(X_train, Y_train, num_words, max_log_length):
    model = Sequential()
    model.add(Embedding(num_words, 32, input_length=max_log_length))
    model.add(Dropout(0.5))
    model.add(LSTM(64, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    # Train the model
    model.fit(X_train, Y_train, validation_split=0.25, epochs=num_epochs, batch_size=batch_size)

    # Evaluate
    score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Model Accuracy: {:0.2f}%".format(acc * 100))

    # Save model
    model.save_weights('output/lstm-weights.h5')
    model.save('output/lstm-model.h5')
    with open('output/lstm-model.json', 'w') as outfile:
        outfile.write(model.to_json())



if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/access.csv'
    
    
    X_train, Y_train, X_test, Y_test, num_words, max_log_length = preprocess(csv_file)
    train(X_train, Y_train, num_words, max_log_length)
