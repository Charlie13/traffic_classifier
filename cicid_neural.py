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
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback

import matplotlib.pyplot as plt


#num_features = 82
num_features = 35
num_classes = 2
batch_size = 128
num_epochs = 50
train_size_per = .8
time_step = 50

# used for ddos1_batch_128 and LSTM
batch_size_train = 1000
batch_size_test = 128


outputDir = 'output/CICID'
#outputDir = 'output/CICID/LSTM'

# give 92%
def create_model():
    model = Sequential()
    model.add(Dense(units=10, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dense(units=190, activation='relu'))
    model.add(Dense(units=80))
    #model.add(Dense(units=50, activation='relu'))
    #model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(units=num_classes, activation='softmax'))
    #model.add(Activation("relu"))
    #model.add(Dense(units=N_CLASS))
    #model.add(Activation("softmax"))

    # choose optimizer and loss function
    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    #opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #opt = optimizers.SGD(lr=0.001)
    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=sgd)

    return model

# around 74% (not good)
def create_model_LSTM():
    model = Sequential()
    #model.add(Dense(units=100, input_dim=X_train.shape[1]))
    model.add(LSTM(128, input_shape=((num_features+1), 1), return_sequences=True ))  
    model.add(LSTM(128, recurrent_dropout=0.5))  
    #model.add(BatchNormalization())
    #model.add(Dense(units=150, activation='relu'))
    #odel.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    # choose optimizer and loss function
    opt = optimizers.SGD(lr=0.001)
    #opt = optimizers.Adam(lr=0.001)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=sgd)

    return model


"""
Step 1 : Load data
"""
def read_data_from_csv(csv_file):
    dataframe = pandas.read_csv(csv_file)
    dataframe.set_value(dataframe[' Label']=='BENIGN',[' Label'],0)
    dataframe.set_value(dataframe[' Label']=='DDoS',[' Label'],1)

    dataset = dataframe.sample(frac=1).values

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

    X = dataset[:, :num_features]
    Y = dataset[:, num_features]

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
          (dataset[:, 3:18]).astype('float32'),
          (dataset[:, 18:20]).astype('float32'),
          (dataset[:, 20:num_features]).astype('float32')
        ), axis=1)
    """

    X_processed = dataset.astype('float32')
    
    print("Features shape: {}".format(X_processed.shape))

    Y = to_categorical(Y, num_classes=num_classes)

    # Divide to train dataset
    train_size = int(len(dataset) * train_size_per)
    X_train = X_processed[0:train_size]
    Y_train = Y[0:train_size]
    # and test dataset
    X_test = X_processed[train_size:len(X_processed)]
    Y_test = Y[train_size:len(Y)]

    return X_train, Y_train, X_test, Y_test



"""
Step 2: Train classifier
"""
def train(X_train, Y_train):
    print("X_train shape: {}".format(X_train.shape))

    
    model = create_model()
    """
    # LSTM requires reshape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = create_model_LSTM()
    """
    print(model.summary())
    
    # Checkpoint
    filepath = outputDir+"/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # Train the model
    model_history = model.fit(X_train, Y_train, validation_split=0.25, epochs=num_epochs, batch_size=batch_size, callbacks=[checkpoint])
    #model_history = model.fit(X_train, Y_train, epochs=num_epochs, callbacks=[plot_losses], batch_size=batch_size)

    # Save model
    weight_file = '{}/lstm_weights.h5'.format(outputDir)
    model_file = '{}/lstm_model.h5'.format(outputDir)
    model.save_weights(weight_file)
    model.save(model_file)

    return model_history, model


"""
Step 3: Evaluate model
"""
def evaluate(model, X_test, Y_test):
    # Evaluate
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print("\nLoss: {}".format(score))
    print("Accuracy: {:0.2f}%".format(acc * 100))


"""
Plot training history
"""
def plot_model(history):
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
"""
Plot during train
"""
class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        #clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()

plot_losses = PlotLosses()



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
    #print(Y_train)
    model_history, model = train(X_train, Y_train)
    evaluate(model, X_test, Y_test)
    #plot_model(model_history)
