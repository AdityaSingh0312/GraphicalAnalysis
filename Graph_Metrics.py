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

cryptopunks_bidsold['log_eth'] = np.log(cryptopunks_bidsold['eth'] + np.min(cryptopunks_bidsold[cryptopunks_bidsold['eth']>0]['eth']))
stdscaler = StandardScaler()
cryptopunks_bidsold['log_norm_eth'] = stdscaler.fit_transform(cryptopunks_bidsold['log_eth'].values.reshape(-1,1))fig = px.histogram(cryptopunks_bidsold, x='eth', marginal='rug', title='Histogram of Ethereum for CryptoPunks Bid & Sold', height=600)

fig = px.histogram(cryptopunks_bidsold, x='eth', marginal='rug', title='Histogram of Ethereum for CryptoPunks Bid & Sold', height=600)
fig.show()

fig = px.histogram(cryptopunks_bidsold, x='log_norm_eth', marginal='rug', title='Histogram of Log Normalized Ethereum for CryptoPunks Bid & Sold', height=600)
fig.show()



def GetGraphMetrics(graph):
    
    graph_degree = dict(graph.degree)
    print("Graph Summary:")
    print(f"Number of nodes : {len(graph.nodes)}")
    print(f"Number of edges : {len(graph.edges)}")
    print(f"Maximum degree : {np.max(list(graph_degree.values()))}")
    print(f"Minimum degree : {np.min(list(graph_degree.values()))}")
    print(f"Average degree : {np.mean(list(graph_degree.values()))}")
    print(f"Median degree : {np.median(list(graph_degree.values()))}")
    print("")
    print("Graph Connectivity")
    try:
        print(f"Connected Components : {nx.number_connected_components(graph)}")
    except:
        print(f"Strongly Connected Components : {nx.number_strongly_connected_components(graph)}")
        print(f"Weakly Connected Components : {nx.number_weakly_connected_components(graph)}")
    print("")
    print("Graph Distance")
    try:
        print(f"Average Distance : {nx.average_shortest_path_length(graph)}")
        print(f"Diameter : {nx.algorithms.distance_measures.diameter(graph)}")
    except:
        shortest_lengths = []
        for C in nx.strongly_connected_components(graph):
            shortest_lengths.append(nx.average_shortest_path_length(G.subgraph(C)))
        print(f"Average Shortest Lengths of Strongly Connected Components : {np.mean(shortest_lengths)}")
    print("")
    print("Graph Clustering")
    print(f"Transitivity : {nx.transitivity(graph)}")
    print(f"Average Clustering Coefficient : {nx.average_clustering(graph)}")
    
    return None
  
G = nx.from_pandas_edgelist(cryptopunks_bidsold, 'from', 'to', ['punk_id', 'eth', 'log_norm_eth', 'txn_type', 'platform', 'accessories', 'txn_type_color'], create_using=nx.DiGraph())
d = dict(G.degree)

k = G.subgraph(np.array(np.random.choice(list(G.nodes), int(len(list(G.nodes))/2))))
pos = nx.kamada_kawai_layout(k)

e = nxa.draw_networkx_edges(k, pos=pos)  # get the edge layer
n = nxa.draw_networkx_nodes(k, pos=pos)  # get the node layer

n = n.mark_circle().encode().interactive()
e = e.mark_line().encode(
        color=alt.Color('eth'),
        stroke=alt.Stroke('eth')
        )

(e+n).properties(width=1600,height=800, title='CryptoPunks Network (subset)')



degree_freq = np.array(nx.degree_histogram(G)).astype('float')
log_degree_freq = np.log(degree_freq)

fig = px.histogram(degree_freq, color_discrete_sequence=['salmon'], marginal='rug', 
                   title='Histogram of Degree Frequency for CryptoPunks Bid & Sold', height=600)
fig.show()

fig = px.histogram(log_degree_freq, color_discrete_sequence=['indianred'], marginal='rug', 
                   title='Histogram of Log Degree Frequency for CryptoPunks Bid & Sold', height=600)
fig.show()



