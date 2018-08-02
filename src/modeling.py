# -*- coding: utf-8 -*-
import numpy as np
from dataset_handler import load_dataset
from dataset_handler import write_to_csv
import os

from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from collections import defaultdict
from sklearn.metrics import confusion_matrix

np.random.seed(1337)  # for reproducibility

originalclass = []
predictedclass = []
nn_layers = 32


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(nn_layers,), random_state=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
    ]


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


def voting_classifier():
    clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(nn_layers,), random_state=1)
    clf2 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    clf3 = SVC(gamma=2, C=1, probability=True)
    clf4 = AdaBoostClassifier()

    eclf = VotingClassifier(estimators=[('NN', clf1), ('rf', clf2), ('ADA', clf4)],
        voting = 'soft', weights = [1, 1,1],
        flatten_transform = True)
    return  eclf

def run_all_classifiers():
    X, y = get_classification()
    scores = []
    for name, clf in zip(names, classifiers):
        score, report = cross_validate(clf, 10)
        scores.append([name, score.mean()])
        print ("%s score: %s" % (name, score.mean()) )
        print (report)
        #print_report_csv(report, name)
    print_scores_csv(scores)


def print_report_csv(report, name):
    report_dict = report2dict(report)
    data = []
    y = ["english", "spanish", "arabic", "mandarin", "french", "average"]
    data.append(['language', 'Precision', 'Recall', 'f1-score', 'support'])
    i = 0
    for key, language in report_dict.items():
        stats = []
        stats.append(y[i])
        i+=1
        stats.append(language["precision"] * 100)
        stats.append(language["recall"] * 100)
        stats.append(language["f1-score"] * 100)
        stats.append(language["support"])
        data.append(stats)
    write_to_csv('../results/' + name + ".csv", data)

def print_scores_csv(scores):
    write_to_csv('../results/scores.csv', scores)

def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data

def class_feature_importance(X, Y, feature_importances):
    N, M = X.shape
    X = scale(X)

    out = {}
    for c in set(Y):
        out[c] = dict(
            zip(range(N), np.mean(X[Y==c, :], axis=0)*feature_importances)
        )

    return out

def cross_validate(clf, n):
    originalclass.clear()
    predictedclass.clear()
    X, y = get_classification()
    score = cross_val_score(clf, X, y, cv=n, scoring=make_scorer(classification_report_with_accuracy_score))
    report = classification_report(originalclass, predictedclass)
    cnf_matrix = confusion_matrix(originalclass, predictedclass)
    return score, report, cnf_matrix

def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    #print(classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

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


def neural_net(hdden_layers):
    X, y = get_classification()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hdden_layers,), random_state=1)
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


from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

batch_size = 2
hidden_units = 15
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
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1000, validation_data=(X_test, Y_test))
    X_test = X_test[:-1, :, :]
    y_test = y_test[:-1]
    y_test = [ int(x) for x in y_test ]
    y_pred = model.predict_classes(X_test, batch_size=batch_size)
    report = classification_report(y_test, y_pred)
    print(report)
    print_report_csv(report, "Recurrent_Neural_Network")
