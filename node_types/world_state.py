from gqlalchemy import Memgraph, Node, Relationship, match
from gqlalchemy import Node
from node_types.conversation import Message

from utils.enums import MemoryRelations
from typing import List, Optional


# How to handle world state updates? Each world state node is permuted,
# with the new state reflected as a new world state graph, with an ACTION relation.
# To some other world state node. Previous states are implicitly in scope via the
# Agents memory, so for now, not explicitly creating a timeseries graph of all the states
# (Tech Debt, & R&D Assignment).
# Use betweenness centrality to rank importance of nodes in the world state & focus attention
# Reflect on state each step. Agent has a tool it can use to query the DB, eg Semantic DB if it
# wants more information on some node
class WorldStateAction(Relationship, type=MemoryRelations.WorldStateAction):
    action: Optional[str]

class WorldState(Node):
    id: Optional[str]
    name: Optional[str]
    
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

    # TODO: Add Document Type
    def get_knowledge_sources(self) -> List[Message]:
        query = (
            match()
            .node(node=self, variable="s")
            .to(MemoryRelations.KnowledgeSource, variable="r")
            .node(variable="m")
            .return_("m")
            .execute()
        )
        results = [Message(**result["m"]) for result in list(query)]
        return results

    def get_semantic_meaning(self) -> List[Message]:
        query = (
            match()
            .node(node=self, variable="s")
            .to(MemoryRelations.SemanticMeaning, variable="r")
            .node(variable="m")
            .return_("m")
            .execute()
        )
        results = [Message(**result["m"]) for result in list(query)]
        return results




db = Memgraph()

class Relation(Relationship, type="RELATION"):
    type: str

def get_node(name):
    try:
        node = WorldState(name=name).load(db=db)
    except:
        node = None
    return node

def store_world_state(triplets):
    # Store the triplets in the graph
    for triplet in triplets:
        # Check if WorldState already exists
        try:
            subject = WorldState(name=triplet['head']).load(db=db)
        except:
            subject = WorldState(name=triplet['head']).save(db=db)
        try:
            object = WorldState(name=triplet['tail']).load(db=db)
        except:
            object = WorldState(name=triplet['tail']).save(db=db)
        Relation(
            _start_node_id=subject._id,
            _end_node_id=object._id,
            type=triplet['type']
        ).save(db)