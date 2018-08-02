#!/usr/bin/env python

from dataset_handler import load_all_wav_into_csv
from dataset_handler import create_dataset
from dataset_handler import load_dataset


from modeling import random_forest
from modeling import decision_tree
from modeling import cross_validate
from modeling import support_vector_machine
from modeling import naive_bayes_mn
from modeling import rnn
from modeling import neural_net
from modeling import get_classification
from modeling import run_all_classifiers
from modeling import class_feature_importance
from modeling import voting_classifier
import numpy as np
import visualisation as v


####### BUILDING THE DATASET FROME .WAV #######
# load_all_wav_into_csv()
#create_dataset()
###############################################


#run_all_classifiers()


# clf = voting_classifier()
# score, report = cross_validate(clf, 10)
# print(score.mean())
# print(report)

# random_forest = random_forest()
# score, report = cross_validate(random_forest, 10)
#
# X, y = get_classification()
# random_forest.fit(X, y)
# v.build_feature_importance_histogram(random_forest.feature_importances_)
# cfi = class_feature_importance(X,y,random_forest.feature_importances_)
# v.build_feature_importance_histogram(cfi[str(0)], 'english')
# v.build_feature_importance_histogram(cfi[str(1)], 'spanish')
# v.build_feature_importance_histogram(cfi[str(2)], 'arabic')
# v.build_feature_importance_histogram(cfi[str(3)], 'mandarin')
# v.build_feature_importance_histogram(cfi[str(4)], 'french')

# print(score.mean())
# print(report)
#
# dt = decision_tree()
# score, report = cross_validate(dt, 10)
# print(score.mean())
# print(report)

# maxscore = 0
# best_layer = 0
# for layers in range(60):
#     print(layers+1)
#     n_net = neural_net(layers+1)
#     score, report = cross_validate(n_net, 10)
#     print(score.mean())
#     if score.mean() > maxscore:
#         maxscore = score.mean()
#         best_layer = layers+1
#
# print(best_layer)
import matplotlib.pyplot as plt

n_net = neural_net(32)
score, report, cnf_matrix = cross_validate(n_net, 10)

np.set_printoptions(precision=2)
plt.figure()
v.plot_confusion_matrix(cnf_matrix, classes=["anglais", "espagnol", "arabe", "mandarin", "francais"], title='Confusion matrix, without normalization')

plt.figure()
v.plot_confusion_matrix(cnf_matrix, classes=["anglais", "espagnol", "arabe", "mandarin", "francais"], normalize=True,
                      title='Matrice de confusion normalisée')

plt.show()

# print(score.mean())
# print(report)
#
# v.build_heatmap(report)
#
# svm = support_vector_machine()
# score, report = cross_validate(svm, 10)
# print(score.mean())
# print(report)
#
# naive_bayes = naive_bayes_mn()
# score, report = cross_validate(naive_bayes, 10)
# print(score.mean())
# print(report)


#rnn()
