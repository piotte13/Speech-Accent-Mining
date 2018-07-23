#!/usr/bin/env python

from helpers import load_all_wav_into_csv
from helpers import create_dataset
from helpers import load_dataset


from core import random_forest
from core import decision_tree
from core import cross_validate
from core import support_vector_machine
from core import naive_bayes_mn
from core import rnn
from core import neural_net


####### BUILDING THE DATASET FROME .WAV #######
#load_all_wav_into_csv()
#create_dataset()
###############################################


random_forest = random_forest()
score = cross_validate(random_forest, 10)
print("Random Forest mean: " +  str(score.mean()))

# dt = decision_tree()
# score = cross_validate(dt, 10)
# print("Decision Tree Classifier mean: " + str(score.mean()))

neural_net = neural_net()
score = cross_validate(neural_net, 10)
print("Neural Net: " + str(score.mean()))


# svm = support_vector_machine()
# score = cross_validate(svm, 10)
# print("Support Vector Machine mean: " + score.mean())


# naive_bayes = naive_bayes_mn()
# score = cross_validate(naive_bayes, 10)
# print("Naive bayes multinomial: " + score.mean())


#rnn()
