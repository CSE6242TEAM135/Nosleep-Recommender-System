# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:45:07 2020

@author: karin
"""
import networkx as nx
import plotly.graph_objs as go
from NoSleepRecommender.controllers.DynamoDBAPI import DynamoDBAPI


def plotly_NetworkGraph(stories):
    db = DynamoDBAPI()
    dod = {}
    # first add the connections for the selected story
    current_story_id = stories['current_story']['id']
    ranked_list = stories["ranked_list"]
    dod[current_story_id] = ranked_list

    # now add other connected stories
    recommended_id_list = stories['recommendations']['story_id'].tolist()  # list of recommended ids
    recommended2_id_list = stories['recommendations'][
        'recommendations'].tolist()  # list of lists: recommended ids for each recommended story
    recommended2_scores_list = stories['recommendations']['recommendations_scores'].tolist()

    for rec_id in recommended_id_list:
        stories2 = db.get_recommendations(rec_id)
        dod[rec_id] = stories2["ranked_list"]

    # for i in range(len(recommended_id_list)):
    #     current_story_id = recommended_id_list[i]
    #     current_recommended_list = recommended2_id_list[i]
    #     current_scores_list = recommended2_scores_list[i] #not standardized
    #     list_num = [float(i) for i in current_scores_list]
    #     current_scores_list = minmax_scale(scores_num)
    #     ranked_list = {current_recommended_list[j]: {'score':current_scores_list[j]} for j in range(len(current_recommended_list))}
    #     dod[current_story_id] = ranked_list

    G = nx.Graph(dod)
    pos = nx.drawing.layout.spring_layout(G)

    ## nodes
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    unique_weights = list(set([edge_attr['score'] for (node1, node2, edge_attr) in G.edges(data=True)]))

    for weight in unique_weights:
        weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in G.edges(data=True) if
                          edge_attr['score'] == weight]
        width = weight * 100
        nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, width=width)

    trace_list = []
    for weight in unique_weights:
        edge_x = []
        edge_y = []
        for (node1, node2, edge_attr) in G.edges(data=True):
            if edge_attr['score'] == weight:
                x0, y0 = G.nodes[node1]['pos']
                x1, y1 = G.nodes[node2]['pos']
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
                width = weight * 5
                trace_list.append(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=width, color='#888'),
                    hoverinfo='none',
                    mode='lines'))

    nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, width=width)

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
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
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
        node_text.append(adjacencies[0] + ',# of stories: ' + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=trace_list,
                    layout=go.Layout(
                        title='<br>Network Graph for Recommended Stories',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig.to_html(full_html=False)