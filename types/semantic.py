# Semantic Database facts
import urllib

from gqlalchemy import Memgraph, Node, Relationship, match
from uuid import uuid4 as uuid
from pydantic import BaseModel, Field
from typing import List, Optional
from urllib.parse import unquote
from gqlalchemy.query_builders.memgraph_query_builder import Order

from utils.enums import MemoryRelations
from .conversation import Message
# Semantic Database facts

db = Memgraph()

# Relates a Semantic node to a Knowledge Source node (eg, a Message from a user or Document from a website)
class KnowledgeSource(Relationship, type=MemoryRelations.KnowledgeSource):
    pass

class SemanticMeaning(Relationship, type=MemoryRelations.SemanticMeaning):
    pass

class Semantic(Node):
    id: Optional[str]
    name: Optional[str]
    description: Optional[str]
    source: Optional[str]
    embedding: Optional[list[float]]

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
        # Convert to list of Messages, removing them from the dict
        results = [Message(**result["m"]) for result in list(query)]
        return results
