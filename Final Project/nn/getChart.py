import plotly
import plotly.offline as py
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import pyplot

class plotter:

    def __init__():
        print('init plotter')

    def plot(testX, testY, prediction):
            
        scatter0 = go.Scatter(x=testX, y=testY, name= 'Actual',
                           line = dict(color = ('rgb(255, 102, 102)'),width = 1))
        scatter1 = go.Scatter(x=testX, y=prediction, name= 'Predicted',
                           line = dict(color = ('rgb(0, 155, 0)'),width = 1))
        data = [scatter0, scatter1]
        layout = dict(title = 'Bitcoin price',
                     xaxis = dict(title = 'Date'), yaxis = dict(title = 'Dollars'))
        fig = dict(data=data, layout=layout)
        py.iplot(fig, filename='prices')
