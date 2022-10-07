import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from pylab import rcParams
import operator
import random

import networkx as nx
import nx_altair as nxa
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import ndlib.models.opinions as op
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence
from ndlib.viz.mpl.PrevalenceComparison import DiffusionPrevalenceComparison

from Graph_Metrics import GetGraphMetrics

RANDOM_SEED=0

alt.data_transformers.disable_max_rows()

cryptopunks = pd.read_json('txn_history-2021-10-07.jsonl', lines=True)
cryptopunks.shape
# (167492, 12)

cryptopunks_bidsold = cryptopunks[cryptopunks['txn_type'].isin(['Bid', 'Sold'])].sort_values("timestamp")
cryptopunks_bidsold.shape
# (45787, 12)

cryptopunks_bidsold['to'] = np.where(cryptopunks_bidsold['to']=='', np.nan, cryptopunks_bidsold['to'])
cryptopunks_bidsold['to'] = cryptopunks_bidsold['to'].fillna(method='ffill')
first_holder = cryptopunks_bidsold[cryptopunks_bidsold['txn_type']=='Sold'].iloc[0, 1]
cryptopunks_bidsold['to'] = np.where(cryptopunks_bidsold['to'].isna(), first_holder, cryptopunks_bidsold['to'])
cryptopunks_bidsold['txn_type_color'] = np.where(cryptopunks_bidsold['txn_type']=='Bid', '#40E0D0', '#8C000F')
cryptopunks_bidsold = cryptopunks_bidsold.rename(columns={'source': 'platform'})
cryptopunks_bidsold


def plot_deg_distribution(G, z=-1, log=False):

    freq = nx.degree_histogram(G);
    x = [i for i in range(len(freq))]
    
    if log:
        freq = [np.log(f+1) for f in freq]
        x = [np.log(i+1) for i in x]
        fig = px.scatter(x=x, y=freq, color_discrete_sequence=['indianred'], title='Log-Log Scale of Degree Frequency', height=600)
        fig.show()

        return None
    
    if z < 0:
        fig = px.scatter(x=x, y=freq, color_discrete_sequence=['indianred'], title='Linear Scale of Degree Frequency', height=600)
        fig.show()

    else:
        fig = px.scatter(x=x[:20], y=freq[:20], color_discrete_sequence=['indianred'], title='Zoom-In View of Degree Frequency', height=600)
        fig.show()

    return None
  
plot_deg_distribution(G, log=True)


random.seed(RANDOM_SEED)
def generativeModels(random_model):
    tran = []  # stores transitivity values
    clus = []  # stores average clustering values

    for i in range(0,1000):
        if random_model == 'erdos_renyi':
            g = nx.fast_gnp_random_graph(n, p, directed=True)
        elif random_model == 'preferential_attachment':
            g = nx.barabasi_albert_graph(n, avg_m)
        elif random_model == 'small_world':
            g = nx.watts_strogatz_graph(n, 4, .1)
            
        tran.append(nx.transitivity(g))
        clus.append(nx.average_clustering(g))
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Transitivity', 'Average Clustering'), specs=[[{'type': 'histogram'}, {'type': 'histogram'}]])

    fig.add_trace(
        go.Histogram(x=np.array(tran), autobinx=True),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(x=np.array(clus), autobinx=True),
        row=1, col=2
    )
    fig.show()

n = len(G.nodes)  # number of nodes
m = len(G.edges)  # number of edges
p = (2*m)/(n*(n-1))  # probability of edges between nodes
G1 = nx.fast_gnp_random_graph(n, p, directed=False, seed=RANDOM_SEED)

plot_deg_distribution(G1, log=True);
generativeModels('erdos_renyi')


avg_m = int(m/n)  # average number of nodes increased by each additional node
G2 = nx.barabasi_albert_graph(n, avg_m, seed=RANDOM_SEED)

plot_deg_distribution(G2, log=True);
generativeModels('preferential_attachment')



G3 = nx.watts_strogatz_graph(n,11,p=.1, seed=RANDOM_SEED)

plot_deg_distribution(G3, log=True);
generativeModels('small_world')