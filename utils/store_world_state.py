from typing import Optional

from gqlalchemy import Memgraph, Node, Relationship, match
from gqlalchemy import Node

db = Memgraph()
class Entity(Node):
    name: Optional[str]
    # TODO: Add embeddings here & example of use in chatbot
    embeddings: Optional[list]
    description: Optional[str]

    def get_relations(self):
        # Get relations here a few nodes. Note we could train a model to convert natural language
        # to cypher queries here, let the agent models drive everything including storage
        # and retrieval of data.
        query_template = """
        MATCH (n {{name: '{0}'}})-[r1*1..2]-(connectedNodes)-[r2*1..2]-(nextNodes)
        RETURN n, r1, connectedNodes, r2, nextNodes
        """
        query = query_template.format(self.name)
        result = db.execute_and_fetch(query)
        return list(result)

class Relation(Relationship, type="RELATION"):
    type: str

def get_node(name):
    try:
        node = Entity(name=name).load(db=db)
    except:
        node = None
    return node

def store_world_state(triplets):
    # Store the triplets in the graph
    for triplet in triplets:
        # Check if Entity already exists
        try:
            subject = Entity(name=triplet['head']).load(db=db)
        except:
            subject = Entity(name=triplet['head']).save(db=db)
        try:
            object = Entity(name=triplet['tail']).load(db=db)
        except:
            object = Entity(name=triplet['tail']).save(db=db)
        Relation(
            _start_node_id=subject._id,
            _end_node_id=object._id,
            type=triplet['type']
        ).save(db)