# -*- coding: utf-8 -*-
from helpers import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier


def cross_validate(clf, n):
    dataset = load_dataset()
    X, y = get_classification()
    return cross_val_score(clf, X, y, cv=n)

def random_forest():
    X, y = get_classification()
    clf = RandomForestClassifier(random_state=0)
    #clf.fit(X, y)
    return clf

def support_vector_machine():
    X, y = get_classification()
    clf = LinearSVC(random_state=0)
    #clf.fit(X, y)
    return clf

def decision_tree():
    X, y = get_classification()
    clf = DecisionTreeClassifier(random_state=0)
    #clf.fit(X, y)
    return clf

def get_classification():
    dataset = load_dataset()
    X = dataset[:,:-1]
    y = dataset[:, -1]
    # X, y = make_classification(n_samples=len(dataset[:, 0]), n_features=len(dataset[0, :]),
    #                            n_informative=len(dataset[0, :]) - 1, n_redundant=0,
    #                            random_state=0, shuffle=False, n_classes=214)
    return X, y

from sklearn.cluster import DBSCAN
from sklearn import metrics


def dbscan():
    X, y = get_classification()
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(y, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(y, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()