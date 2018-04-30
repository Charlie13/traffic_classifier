import sys
import os
import json
import pandas
import numpy
import optparse
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
import scipy as sp

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.layers.normalization import BatchNormalization


N_FEATURES = 41
N_CLASS = 24
batch_size = 32
num_epochs = 3


"""
Step 1 : Load and preprocess data
"""
def preprocess(csv_file):
    dataset = pandas.read_csv(csv_file)
    #dataset = dataframe.sample(frac=1).values

    #tokenized = map(lambda msg, ft1, ft2: features([msg,ft1,ft2]), dataset.message, dataset.feature_1, dataset.feature_2)

    """
    tokenized = map(lambda h2, h3, h4: 
                    features([h2, h3, h4]), 
                    dataset['h2'], dataset['h3'], dataset['h4']
                )
    """
    """
    mapper = DataFrameMapper([
        (['h1', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24', 'h25', 'h26', 'h27', 'h28', 'h29', 'h30', 'h31', 'h32', 'h33', 'h34', 'h35', 'h36', 'h37', 'h38', 'h39', 'h40', 'h41'], None),
        #('message',CountVectorizer(binary=True, ngram_range=(1, 2)))
        ('h2', CountVectorizer(binary=True, ngram_range=(1, 2))),
        ('h3', CountVectorizer(binary=True, ngram_range=(1, 2))),
        ('h4', CountVectorizer(binary=True, ngram_range=(1, 2)))
    ])
    dataset = mapper.fit_transform(dataset)
    """

    dataset = dataset.apply(pandas.to_numeric)
    dataset = dataset.as_matrix()
    print(dataset)

    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    #y = dataset[41].values.astype(numpy.float32) 

    dataset = sp.sparse.hstack((vectorizer.fit_transform(dataset['h2']), dataset[['h1', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24', 'h25', 'h26', 'h27', 'h28', 'h29', 'h30', 'h31', 'h32', 'h33', 'h34', 'h35', 'h36', 'h37', 'h38', 'h39', 'h40', 'h41', 'h42']].values), format='csr')
    #X_columns = vectorizer.get_feature_names()+dataset[['h1', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24', 'h25', 'h26', 'h27', 'h28', 'h29', 'h30', 'h31', 'h32', 'h33', 'h34', 'h35', 'h36', 'h37', 'h38', 'h39', 'h40', 'h41', 'h42']].columns.tolist()

    X = dataset[:, :N_FEATURES]
    Y = dataset[:, N_FEATURES]


    return X, Y


"""
Step 2: Build and train model
"""
def train(X_train, Y_train):
    model = Sequential()
    #model.add(Dense(output_dim=64, input_dim=N_FEATURES))
    model.add(BatchNormalization())
    model.add(LSTM(64, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(output_dim=N_CLASS, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    # Train the model
    model.fit(X_train, Y_train, validation_split=0.25, epochs=num_epochs, batch_size=batch_size)

    # Evaluate
    score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Model Accuracy: {:0.2f}%".format(acc * 100))

    # Save model
    model.save_weights('output/flow_lstm-weights.h5')
    model.save('output/flow_lstm-model.h5')
    with open('output/flow_lstm-model.json', 'w') as outfile:
        outfile.write(model.to_json())


"""
Step 3: Evaluate
"""
def evaluate(X_test, Y_test):
    # Load model
    model = load_model('output/flow_lstm-model.h5')

    # Evaluate
    score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Model Accuracy: {:0.2f}%".format(acc * 100))


"""
Step 4: Predict
"""
def predict(X_test, Y_test):
    # Load and compile model
    model = load_model('output/flow_lstm-model.h5')
    model.load_weights('output/flow_lstm-weights.h5')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Predict
    #prediction = model.predict(log_entry_processed)
    #print prediction[0]

    classes = model.predict_classes(X_test, batch_size=batch_size)
    accuration = numpy.sum(classes == Y_test)/float(len(Y_test)) * 100

    print "Test Accuration : " + str(accuration) + '%'
    print "Prediction :"
    print classes
    print "Target :"
    print numpy.asarray(Y_test, dtype="int32")


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/KDD.csv'
    
    X_train, Y_train = preprocess('data/KDDTrain+.csv')
    #X_test, Y_test = preprocess('data/KDDTest+.csv')

    #train(X_train, Y_train, num_words, max_log_length)
