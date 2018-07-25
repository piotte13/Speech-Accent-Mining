import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from modeling import report2dict

plotly.tools.set_credentials_file(username='piotte13', api_key='ZQVdwlZvh6zPgnySWuE6')




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