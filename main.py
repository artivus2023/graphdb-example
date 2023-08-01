from ctransformers import AutoModelForCausalLM
from agent.chat import respond_to
from agent.describe_world_state import describe_world_state
from agent.summarize import summarize
from utils.enums import ConversationRole
from utils.memgraph_to_networkx import memgraph_to_networkx
from types.world_state import store_world_state, get_node
from utils.triplet_extractor import extract_triplets, triplets_to_networkx
from agent.conversation_memory import ConversationMemory
import networkx as nx
# main.py
from utils.graph import draw_graph

def initialize_model():
    return AutoModelForCausalLM.from_pretrained(
        'models/nous-hermes-llama-2-7b.ggmlv3.q2_K.bin',
        max_new_tokens=2048,
        model_type='llama',
        stream=True,
        temperature=0.1,
        stop=['\n', 'User: ', '### Example ']
    )

def handle_user_input(user_input, history, llm, conversation_memory):
    # Extract graph from user input
    extracted_triplets = extract_triplets(user_input)
    store_world_state(extracted_triplets)

    # Update conversation history and construct summary
    history = respond_to(user_input, history, llm)
    
    # Add user input to conversation graph
    conversation_memory.add_message(ConversationRole.USER, user_input)
    
    return history, extracted_triplets

def main(history):
    llm = initialize_model()
    conversation_memory = ConversationMemory(name="repl")
    
    # Load conversation history
    messages = conversation_memory.conversation.get_messages()
    history = [(message["m"].role, message["m"].content) for message in messages]

    while True:
        user_input = input("User: ")
        history, extracted_triplets = handle_user_input(user_input, history, llm, conversation_memory)

        # Add to conversation graph
        summary = summarize(history, llm)

        # Extract graph from summary
        summary_triplets = extract_triplets(summary)
        store_world_state(summary_triplets)
        summary_graph = triplets_to_networkx(summary_triplets)

        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(summary_graph, endpoints=True)
        # Find the node with the highest betweenness centrality
        # This will be the most important thing in the conversation right now
        # Could do a lot of other kinds of graph analysis here too for context
        # like using graph neural networks to find missing or important context
        # Other similar patterns of knowledge, whatevaaaaa
        central_node = max(centrality, key=centrality.get)

        # Fetch the central node for this interaction from the database
        entity = get_node(central_node)
        if entity:
            # Get all relations 2 deep for some context (again, lotta wats to walk through
            # it's neighbours and look for relevant stuff for whatever purpose)
            nodes_list = entity.get_relations()
            graph = memgraph_to_networkx(nodes_list)
            draw_graph(graph)
        # Let's draw the conversation graph too

if __name__ == "__main__":
    from collections import deque
    history = deque(maxlen=10)
    main(history)