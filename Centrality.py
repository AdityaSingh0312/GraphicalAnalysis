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






def CreateBarChartForCentralityMeasures(sorted_centrality_users, centrality_name):
    df = pd.DataFrame(sorted_centrality_users, columns=['User', centrality_name])
    return df
  
metric_main_df = pd.DataFrame(index=list(d.keys()))

degree_centrality_df = CreateBarChartForCentralityMeasures(sorted(nx.degree_centrality(G).items(), key=lambda x:x[1], reverse=True), 'Degree Centrality')
closeness_centrality_df = CreateBarChartForCentralityMeasures(sorted(nx.closeness_centrality(G).items(), key=lambda x:x[1], reverse=True), 'Closeness Centrality')
betweenness_centrality_df = CreateBarChartForCentralityMeasures(sorted(nx.betweenness_centrality(G).items(), key=lambda x:x[1], reverse=True), 'Betweenness Centrality')
eigenvector_centrality_df = CreateBarChartForCentralityMeasures(sorted(nx.eigenvector_centrality(G).items(), key=lambda x:x[1], reverse=True), 'Eigenvector Centrality')
pagerank_df = CreateBarChartForCentralityMeasures(sorted(nx.pagerank(G, alpha=.9).items(), key=lambda x:x[1], reverse=True), 'PageRank')

for metric in [degree_centrality_df, closeness_centrality_df, betweenness_centrality_df, eigenvector_centrality_df, pagerank_df]:
    metric = metric.set_index('User')
    metric_main_df = metric_main_df.join(metric)
    
metric_main_df['Log Flow Centrality'] = np.log(metric_main_df['Betweenness Centrality']+np.min(metric_main_df[metric_main_df['Betweenness Centrality']>0]['Betweenness Centrality']))
stdscaler = StandardScaler()
metric_main_df['Normalized Flow Centrality'] = stdscaler.fit_transform(metric_main_df['Log Flow Centrality'].values.reshape(-1,1))

fig = px.histogram(metric_main_df, x='Betweenness Centrality', color_discrete_sequence=['blue'], marginal='rug', 
                   title='Histogram of Betweenness Centrality for Users Bidding & Selling CryptoPunks', height=600)
fig.show()

fig = px.histogram(metric_main_df,  x='Log Flow Centrality', color_discrete_sequence=['teal'], marginal='rug', 
                   title='Histogram of Log Flow Centrality for Users Bidding & Selling CryptoPunks', height=600)
fig.show()

fig = px.histogram(metric_main_df,  x='Normalized Flow Centrality', color_discrete_sequence=['goldenrod'], marginal='rug', 
                   title='Histogram of Normalized Flow Centrality for Users Bidding & Selling CryptoPunks', height=600)
fig.show()



for n in G.nodes():
    G.nodes[n]['name'] = n
    G.nodes[n]['degree'] = d[n]
    G.nodes[n]['degree_centrality'] = metric_main_df[metric_main_df.index==n]['Degree Centrality'].values[0]
    G.nodes[n]['closeness_centrality'] = metric_main_df[metric_main_df.index==n]['Closeness Centrality'].values[0]
    G.nodes[n]['betweenness_centrality'] = metric_main_df[metric_main_df.index==n]['Betweenness Centrality'].values[0]
    G.nodes[n]['log_flow_centrality'] = metric_main_df[metric_main_df.index==n]['Log Flow Centrality'].values[0]
    G.nodes[n]['flow_centrality'] = metric_main_df[metric_main_df.index==n]['Normalized Flow Centrality'].values[0]
    G.nodes[n]['eigenvector_centrality'] = metric_main_df[metric_main_df.index==n]['Eigenvector Centrality'].values[0]
    G.nodes[n]['pagerank'] = metric_main_df[metric_main_df.index==n]['PageRank'].values[0]
    
pos = nx.kamada_kawai_layout(G)

e = nxa.draw_networkx_edges(G, pos=pos)  # get the edge layer
n = nxa.draw_networkx_nodes(G, pos=pos)  # get the node layer

n = n.mark_circle().encode(
        color=alt.Color('flow_centrality:Q', scale=alt.Scale(scheme='inferno')), 
        size=alt.Size('flow_centrality:Q',
                     scale=alt.Scale(range=[10,100])),
        tooltip=[alt.Tooltip('name'), alt.Tooltip('flow_centrality')]
        ).interactive()
e = e.mark_line().encode(
        color=alt.Color('log_norm_eth',
                        legend=None),
        stroke=alt.Stroke('log_norm_eth')
        )

(e+n).properties(width=1500,height=800, title='CryptoPunks Network by Flow Centrality')