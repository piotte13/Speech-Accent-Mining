# -*- coding: utf-8 -*-
import numpy as np
from helpers import load_dataset
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

np.random.seed(1337)  # for reproducibility


def cross_validate(clf, n):
    dataset = load_dataset()
    X, y = get_classification()
    return cross_val_score(clf, X, y, cv=n)


def random_forest():
    X, y = get_classification()
    clf = RandomForestClassifier(random_state=0)
    # clf.fit(X, y)
    return clf


def support_vector_machine():
    X, y = get_classification()
    clf = LinearSVC(random_state=0)
    # clf.fit(X, y)
    return clf


def decision_tree():
    X, y = get_classification()
    clf = DecisionTreeClassifier(random_state=0)
    # clf.fit(X, y)
    return clf


def naive_bayes_mn():
    X, y = get_classification()
    clf = MultinomialNB()
    # clf.fit(X, y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    return clf


def neural_net():
    X, y = get_classification()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,), random_state=1)
    return clf


def get_classification():
    dataset = load_dataset()
    min_max_scaler = preprocessing.MinMaxScaler()
    X = dataset[:, :-1]
    y = dataset[:, -1]
    X = min_max_scaler.fit_transform(X)

    # X, y = make_classification(n_samples=len(dataset[:, 0]), n_features=len(dataset[0, :]),
    #                            n_informative=len(dataset[0, :]) - 1, n_redundant=0,
    #                            random_state=0, shuffle=False, n_classes=214)
    return X, y


# from keras.optimizers import SGD
# from keras.preprocessing import sequence
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.recurrent import LSTM
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import classification_report

batch_size = 2
hidden_units = 10
nb_classes = 6


def rnn():
    X, y = get_classification()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18)
    print('Build model...')

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    X_train = np.reshape(X_train, (int(X_train.shape[0]), 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (int(X_test.shape[0]), 1, X_test.shape[1]))

    # batch_input_shape= (batch_size, X_train.shape[1], X_train.shape[2])

    # note that it is necessary to pass in 3d batch_input_shape if stateful=True
    model.add(LSTM(64, return_sequences=True, stateful=False,
                   batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(64, return_sequences=True, stateful=False))
    model.add(LSTM(64, stateful=False))

    # add dropout to control for overfitting
    model.add(Dropout(.25))

    # squash output onto number of classes in probability space
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    print("Train...")
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=5, validation_data=(X_test, Y_test))
    X_test = X_test[:-1, :, :]
    y_test = y_test[:-1]
    y_pred = model.predict_classes(X_test, batch_size=batch_size)
    print(classification_report(y_test, y_pred))
