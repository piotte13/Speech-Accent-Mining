import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from dataset_handler import write_to_csv
from modeling import report2dict
import numpy as np
import matplotlib.pyplot as plt
import itertools

plotly.tools.set_credentials_file(username='piotte13', api_key='ZQVdwlZvh6zPgnySWuE6')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe réelle')
    plt.xlabel('Classe prédite')

def build_heatmap(report):
    report_dict = report2dict(report)
    data = []

    for key, language in report_dict.items():
        stats = []
        stats.append(language["precision"] * 100)
        stats.append(language["recall"] * 100)
        stats.append(language["f1-score"] * 100)
        #stats.append(language["support"])
        data.append(stats)


    trace = go.Heatmap(z=data,
                       x=['Precision', 'Recall', 'f1-score'],
                       y=["english", "spanish", "arabic", "mandarin", "french", "average"])
    data=[trace]
    py.plot(data, filename='labelled-heatmap')

def build_feature_importance_histogram(importances, name=''):
   data = []
   data.append(["Feature", "Weight"])
   j = 0

   for i in range(13):
       data.append(["average"+str(i+1), importances[j]])
       data.append(["mean"+str(i+1), importances[j+1]])
       data.append(["std"+str(i+1),importances[j+2]])
       data.append(["skew"+str(i+1),importances[j+3]])
       j+=4

   write_to_csv('../results/feature_importance_' + name + '.csv', data)

    # x = []
    #
    # for i in range(13):
    #     x.append("average"+str(i+1))
    #     x.append("mean"+str(i+1))
    #     x.append("std"+str(i+1))
    #     x.append("skew"+str(i+1))
    #
    #
    # trace0 = go.Bar(
    #     x=x,
    #     y=importances,
    #     marker=dict(
    #         color='rgb(158,202,225)',
    #         line=dict(
    #             color='rgb(8,48,107)',
    #             width=1.5,
    #         )
    #     ),
    #     opacity=0.6
    # )
    #
    # data = [trace0]
    # layout = go.Layout(
    #     title='Importance de chacune des features',
    # )
    #
    # fig = go.Figure(data=data, layout=layout)
    #py.plot(fig, filename='feature_importance')