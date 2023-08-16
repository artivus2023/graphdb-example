import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(G):
    # Create a new directed graph
    D = nx.DiGraph()

    # Add nodes and edges with labels to the new graph
    for u, v, data in G.edges(data=True):
        D.add_edge(u, v, label=data['label'])

    # Choose a layout for the graph
    pos = nx.spring_layout(D, k=0.15, iterations=20)  # k regulates the distance between nodes

    # Draw the nodes
    nx.draw_networkx_nodes(D, pos, node_size=500)

    # Draw the edges
    nx.draw_networkx_edges(D, pos)

    # Draw the node labels
    nx.draw_networkx_labels(D, pos, font_size=12)

    # Draw the edge labels
    edge_labels = nx.get_edge_attributes(D, 'label')
    nx.draw_networkx_edge_labels(D, pos, edge_labels=edge_labels)

    plt.show()