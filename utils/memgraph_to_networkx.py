import networkx as nx
import matplotlib.pyplot as plt

# Map Memgraph nodes to NetworkX nodes for easy plotting
def memgraph_to_networkx(nodes_list):
    G = nx.MultiDiGraph()

    for record in nodes_list:
        node1 = record['n']
        node2 = record['connectedNodes']
        next_node = record['nextNodes']

        G.add_node(node1.name, **node1.__dict__)
        G.add_node(node2.name, **node2.__dict__)
        G.add_node(next_node.name, **next_node.__dict__)

        for r1 in record['r1']:
            G.add_edge(node1.name, node2.name, label=r1.properties['type'])

        for r2 in record['r2']:
            G.add_edge(node2.name, next_node.name, label=r2.properties['type'])

    return G
