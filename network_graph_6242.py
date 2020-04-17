# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:45:07 2020

@author: karin
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go
import pandas as pd
from colour import Color
from datetime import datetime
from textwrap import dedent as d
import json
import ast
import pandas as pd
import re
import string
import random
import numpy as np
import os
import math
import string
import json
from itertools import product
from inspect import getsourcefile
from ipywidgets import widgets
import scipy
from sklearn.preprocessing import minmax_scale
import operator


 
## load files 
os.getcwd()
'C:\\Program Files\\PyScripter'
os.chdir(r"C:\\Users\\karin\\Documents\OMSA\\DVA 6242\\Group Project") 

file = open("story_rankings.txt", "r")
recommended_data = file.read()
ranking_dict = ast.literal_eval(recommended_data)
file.close()
       
file = open("story_rankings_scores.txt", "r")
recommended_data = file.read()
scores_dict = ast.literal_eval(recommended_data)
file.close()


dod={}
for key in ranking_dict:
    ranking_list = ranking_dict[key]
    scores_list = scores_dict[key]
    scores_num = [float(i) for i in scores_list] 
    scores_list_std = minmax_scale(scores_num)
    dod[key]={}
    dod[key]={ranking_list[i]:{'score': scores_list_std[i]} for i in range(len(ranking_list))}
    
    
    
## dod= {0: {1:{'weight':1}}}
## dod= {0: {1:{'weight':1},2:{'weight':1}},
##      1: {2:{'weight':3}} }



G1 = nx.Graph(dod)
node_selection = 'abf59b'
node_list= ranking_dict[node_selection][0:10]
node_list.append(node_selection)
G = G1.subgraph(node_list)


edge_x = []
edge_y = []

pos = nx.drawing.layout.spring_layout(G)

## nodes
for node in G.nodes:
    G.nodes[node]['pos'] = list(pos[node])

## edges
# weights=[]
# for edge in G.edges():
#     x0, y0 = G.nodes[edge[0]]['pos']
# ##    print(G.nodes[edge[0]])
#     x1, y1 = G.nodes[edge[1]]['pos']
#     edge_x.append(x0)
#     edge_x.append(x1)
#     edge_x.append(None)
#     edge_y.append(y0)
#     edge_y.append(y1)
#     edge_y.append(None)  
#     try:
#         weights.append(dod[edge[0]][edge[1]]["score"])
#     except KeyError:
#         weights.append(dod[edge[1]][edge[0]]["score"])
 
        
unique_weights = list(set([edge_attr['score'] for (node1,node2,edge_attr) in G.edges(data=True)]))

for weight in unique_weights:
    weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['score']== weight]
    width = weight*100
    nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width)
        
## edge_trace = go.Scatter(
##    x=edge_x, y=edge_y,
##    line=dict(width= 0.5, color='#888'),
##    hoverinfo='none',
#    mode='lines')

trace_list = []
for weight in unique_weights:
    edge_x = []
    edge_y = []
    for (node1,node2,edge_attr) in G.edges(data=True):
        if edge_attr['score']== weight:
            x0, y0 = G.nodes[node1]['pos']
            x1, y1 = G.nodes[node2]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            width = weight*5
            trace_list.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=width, color='#888'),
                hoverinfo='none',
                mode='lines'))
            
nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width)

## textbox = widgets.Dropdown(
##     description='Story:   ',
##     value='abfk8f',
##     options=[a for a in ranking_dict]
## )        

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness = 15 ,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

trace_list.append(node_trace)

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of stories: '+str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=trace_list,
             layout=go.Layout(
                title='<br>Network Graph Recommended Stories',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(

                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )


fig.show()


fig.write_html("graph.html")
