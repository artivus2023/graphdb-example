from ctransformers import AutoModelForCausalLM

from agent.chat import respond_to, history
from agent.describe_world_state import describe_world_state
from agent.summarize import summarize
from utils.memgraph_to_networkx import memgraph_to_networkx
from utils.store_world_state import store_world_state, get_node

from utils.triplet_extractor import extract_triplets
import matplotlib.pyplot as plt
import networkx as nx

# Initialize the model
llm = AutoModelForCausalLM.from_pretrained(
    'models/nous-hermes-llama-2-7b.ggmlv3.q2_K.bin',
    max_new_tokens=2048,
    model_type='llama',
    stream=True,
    temperature=0.1,
    stop=['\n', 'User: ', '### Example ']
)

# Main REPL
def main(history):
    summary = ""
    while True:
        user_input = input("User: ")
        # Get normalised description of input text, we could build a graph off this
        # Get any knowledge from the user input
        # We could do other stuff here too like turn it into a more formal description
        # world_state_description = describe_world_state(user_input, llm)
        # prefix = "Speaker: User, Addressed to: AI, Content: "
        # print(f"World state description: {prefix + world_state_description}")
        extracted_triplets = extract_triplets(user_input)
        print("Got Triplets", extracted_triplets)
        store_world_state(extracted_triplets)
        # print(extracted_triplets)
        for triplet in extracted_triplets:
            print(triplet)
        history = respond_to(user_input, history, llm)
        summary = summarize(history, llm)
        # print(f"Summary: {summary}")

        # Fetch the main entity from the database
        entity = get_node(user_input.split()[0])  # Assuming the first word is the entity name
        if entity:
            # Get all relations 2 deep
            nodes_list = entity.get_relations()
            # for node in nodes_list:
            #     print(node)
            print(nodes_list)
            graph = memgraph_to_networkx(nodes_list)
            print(graph.nodes(data=True))
            draw_graph(graph)

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

if __name__ == '__main__':
    # Start by giving the agent some context, ie your name
    main(history)



