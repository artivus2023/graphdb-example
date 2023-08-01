import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(G):
    # Create a new directed graph
    D = nx.DiGraph()

    # Add nodes and edges with labels to the new graph
    for u, v, data in G.edges(data=True):
        D.add_edge(u, v, label=data['label'])

    pos = nx.shell_layout(D)
    nx.draw(D, pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(D, 'label')
    nx.draw_networkx_edge_labels(D, pos, edge_labels=edge_labels)
    plt.show()
