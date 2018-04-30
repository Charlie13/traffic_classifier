"""
Data used for this script: https://github.com/defcom17/NSL_KDD
Remove the 42th column (name of attack type)
"""

import sys
import os
import json
import pandas
import numpy as np
import optparse

from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

num_features = 41
num_classes = 24
batch_size = 128
num_epochs = 3

"""
Step 1 : Load and preprocess data
"""
def preprocess(csv_file):
    dataframe = pandas.read_csv(csv_file, engine='python', header=None)
    dataset = dataframe.sample(frac=1).values

    #print(dataframe);

    #dataset = dataset.astype(float)

    X = dataset[:, :num_features]
    Y = dataset[:, num_features]


    # Vectorize a text corpus, by turning each text into either a sequence of integers
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X)

    """ Extract and save word dictionary """
    word_dict_file = 'build/nsl-kdd_word-dictionary.json'
    # Create file if not exist
    if not os.path.exists(os.path.dirname(word_dict_file)):
        os.makedirs(os.path.dirname(word_dict_file))
    # Save content
    with open(word_dict_file, 'w') as outfile:
        json.dump(tokenizer.word_index, outfile, ensure_ascii=False)

    num_words = len(tokenizer.word_index)+1
    # Transform all text to a sequence of integers
    X = tokenizer.texts_to_sequences(X)

    """
    X_ft2 = np.array(dataset[:,1]).reshape(-1, 1)
    X_ft3 = np.array(dataset[:,2]).reshape(-1, 1)
    X_ft4 = np.array(dataset[:,3]).reshape(-1, 1)
    X_str = np.concatenate((X_ft2, X_ft3, X_ft4), axis=1)
    print(X_str)
    X = np.concatenate((np.array(dataset[:, 0]).reshape(-1, 1), 
                        X_str, 
                        np.array(dataset[:, 4:40] * 100).astype(int) ), 
                    axis=1)
    print(X)
    """

    """
    num_words += len(np.array(dataset[:, 0]).reshape(-1, 1)) + len(np.array(dataset[:, 4:40] * 100))
    dataset_processed = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
    print(dataset_processed)
    np.save('dataset_processed.npy', dataset_processed)
    """


    max_log_length = 1024
    train_size = int(len(dataset) * .75)

    X_processed = sequence.pad_sequences(X, maxlen=max_log_length, dtype='float32')
    #X_processed = X

    #print(X)
    print(X_processed[0,:])

    Y = to_categorical(Y, num_classes=num_classes)

    # Divide to train dataset
    """
    X_train = X_processed[0:train_size]
    Y_train = Y[0:train_size]
    # and test dataset
    X_test = X_processed[train_size:len(X_processed)]
    Y_test = Y[train_size:len(Y)]
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X_processed, Y, train_size=train_size, random_state=42)

    #print(X_train)
    #print(np.shape(X_train))
    #print(X_test)
    #print(np.shape(X_test))
    #print(X_train[0,:])
    #print(np.shape(X_train[0,:]))

    return X_train, Y_train, X_test, Y_test, num_words, max_log_length


"""
Step 2: Build and train model
"""
def train(X_train, Y_train, num_words, max_log_length):
    # Build the model
    model = Sequential()
    model.add(Embedding(num_words, 32, input_length=max_log_length))
    model.add(Dropout(0.5))
    model.add(LSTM(64, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid')) # TODO: test with ReLU and softmax instead of sigmoid
    opt = optimizers.Adam(lr=0.001) # modify lr or change optimizer if necessary
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())


    # Checkpoint
    filepath = "output/NSL_KDD/lstm-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    #model.fit(X_train, Y_train, validation_split=0.25, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list)
    model.fit(X_train, Y_train, validation_split=0.25, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list)

    # Evaluate
    score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Model Accuracy: {:0.2f}%".format(acc * 100))

    # Save model
    model.save_weights('output/NSL_KDD/lstm-weights.h5')
    model.save('output/NSL_KDD/lstm-model.h5')
    with open('output/NSL_KDD/lstm-model.json', 'w') as outfile:
        outfile.write(model.to_json())


"""
Step 3: Evaluate
"""
def evaluate(X_test, Y_test):
    # Load model
    model = load_model('output/NSL_KDD/lstm-model.h5')

    # Evaluate
    score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Model Accuracy: {:0.2f}%".format(acc * 100))


"""
Step 4: Predict
(TODO)
"""
def predict(X_test, Y_test):
    # Load and compile model
    model = load_model('output/NSL_KDD/lstm-model.h5')
    model.load_weights('output/NSL_KDD/lstm-weights.h5')
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Predict
    #prediction = model.predict(log_entry_processed)
    #print prediction[0]



if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/KDDTrain+.csv'
    
    X_train, Y_train, X_test, Y_test, num_words, max_log_length = preprocess(csv_file)

train(X_train, Y_train, num_words, max_log_length)