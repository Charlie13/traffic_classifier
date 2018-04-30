"""
Data used for this script: https://github.com/defcom17/NSL_KDD
"""

import sys
import os
import json
import pandas
import numpy as np
import optparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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

#num_features = 82
num_features = 35
num_classes = 2
batch_size = 128
num_epochs = 25

outputDir = 'output/CICID/01_05'

"""
Step 1 : Load data
"""
def read_data_from_csv(csv_file):
    dataframe = pandas.read_csv(csv_file)
    dataframe.set_value(dataframe[' Label']=='BENIGN',[' Label'],0)
    dataframe.set_value(dataframe[' Label']=='DDoS',[' Label'],1)

    dataset = dataframe.sample(frac=1).values
    #dataset = dataframe.sample(frac=1).values

    np.save("data/CICID/{}.npy".format(os.path.basename(csv_file)), dataset)

    return dataset

def read_data_from_np(npy_file):
    dataset = np.load(npy_file)
    return dataset

"""
Step 2: Preprocess dataset
"""
def preprocess(dataset):
    print("\nDataset shape: {}".format(dataset.shape))
    print(np.shape(dataset))

    X = dataset[:, :num_features]
    Y = dataset[:, num_features]

    """
    # Vectorize a text corpus, by turning each text into either a sequence of integers
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X)

    # Extract and save word dictionary
    word_dict_file = 'build/CICID/word-dictionary.json'
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

    """
    num_words += len(np.array(dataset[:, 0]).reshape(-1, 1)) + len(np.array(dataset[:, 4:40] * 100))
    dataset_processed = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
    print(dataset_processed)
    np.save('dataset_processed.npy', dataset_processed)

    max_log_length = 1024

    X_processed = sequence.pad_sequences(X, maxlen=max_log_length, dtype='float32')
    """


    """
    X_ft1 = np.array(dataset[:,0]).reshape(-1, 1)
    X_ft3 = np.array(dataset[:,2]).reshape(-1, 1)
    X_str = np.concatenate((X_ft1, X_ft3), axis=1)

    # Vectorize a text corpus, by turning each text into either a sequence of integers
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X_str)
    # Extract and save word dictionary
    word_dict_file = 'build/CICID/word-dictionary.json'
    if not os.path.exists(os.path.dirname(word_dict_file)):
        os.makedirs(os.path.dirname(word_dict_file))
    with open(word_dict_file, 'w') as outfile:
        json.dump(tokenizer.word_index, outfile, ensure_ascii=False)
    # Transform all text to a sequence of integers
    #num_words = len(tokenizer.word_index)+1
    X_str = tokenizer.texts_to_sequences(X_str)

    X_processed = np.concatenate(
        ( np.array(dataset[:, 1]).reshape(-1, 1).astype('float32'), 
          X_str,
          np.array(dataset[:, 3:num_classes]).astype('float32')
        ), axis=1)
    """
    
    X_processed = dataset.astype('float32')
    #print(X)
    #print(X_processed[0,:])

    Y = to_categorical(Y, num_classes=num_classes)

    # Divide to train dataset
    train_size = int(len(dataset) * .75)
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

    return X_train, Y_train, X_test, Y_test



"""
Step 2: Train classifier
"""
def train(X_train, Y_train):
    #print(X_train)
    print(np.shape(X_train))
    #print(Y_train)

    classifier = DecisionTreeClassifier()     # give 100% (????)
    #classifier = KNeighborsClassifier()      # give 94.55%
    classifier.fit(X_train, Y_train)

    return classifier


"""
Step 3: Evaluate classifier
"""
def predict(classifier, X_validation, Y_validation):
    # Evaluate
    predictions = classifier.predict(X_validation)

    print("\n\n")
    print("Prediction: \n{}".format(np.asarray(predictions, dtype="int32")))
    print("Target: \n{}".format(np.asarray(Y_validation, dtype="int32")))
    print(classification_report(Y_validation, predictions))
    print("Test size: {}".format(len(X_validation)))
    print("Test accuration: {}".format(accuracy_score(Y_validation, predictions)))



if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/CICID/ddos1_less.csv'
    
    #dataset = read_data_from_np('{}.npy'.format(csv_file))
    dataset = read_data_from_csv(csv_file)

    X_train, Y_train, X_test, Y_test = preprocess(dataset)
    print("\nTrain samples: {}".format(len(X_train)))
    print("Test samples: {}".format(len(X_test)))

    classifier = train(X_train, Y_train)
    predict(classifier, X_test, Y_test)
