# Semantic Database facts
import urllib

from gqlalchemy import Memgraph, Node, Relationship, match
from uuid import uuid4 as uuid

from pydantic import BaseModel, Field
from typing import List, Optional
from urllib.parse import unquote

from gqlalchemy.query_builders.memgraph_query_builder import Order
from node_types.messages import AIMessage, BaseMessage, FunctionMessage, SystemMessage, UserMessage
from utils.embeddings import create_embedding

from utils.enums import ConversationRole, MemoryRelations, SimulacraNodeType
from utils.helpers import get_timestamp

db = Memgraph()

class Summary(Node):
    id: Optional[str]
    name: Optional[str]
    content: Optional[str]
    embedding: Optional[list[float]]

class Message(Node):
    id: Optional[str]
    timestamp: Optional[int]
    role: Optional[str]
    content: Optional[str]
    embedding: Optional[list[float]]

    def get_agent_message(self, prefix: Optional[str] = None):
        temp_message: BaseMessage | None = None
        content = urllib.parse.unquote(self.content)
        if prefix:
            content = prefix + content
        if self.role == "assistant":
            temp_message = AIMessage(content=content)
        elif self.role == "user":
            temp_message = UserMessage(content=content)
        elif self.role == "system":
            
            temp_message = SystemMessage(content=content)
        elif self.role == "function":
            temp_message = FunctionMessage(content=content)

        return temp_message

    def get_conversation(self):
        query = (
            match()
            .node(labels=SimulacraNodeType.Conversation, variable="c")
            .to(MemoryRelations.HasMessage, variable="r")
            .node(node=self, variable="m")
            .return_("c")
            .execute()
        )
        results = list(query)
        if len(results) == 0:
            return None
        else:
            return Conversation(**results[0]["c"])



class HasSummary(Relationship, type=MemoryRelations.HasSummary):
    pass

class HasSystemMessage(Relationship, type=MemoryRelations.HasSystemMessage):
    pass

class HasMessage(Relationship, type=MemoryRelations.HasMessage):
    pass

class HasReply(Relationship, type=MemoryRelations.HasReply):
    pass
# Conversation
class Conversation(Node):
    id: Optional[str]
    name: Optional[str]
    description: Optional[str]

    def get_summary(self) -> Summary | None:
        query = (
            match()
            .node(node=self, variable="s")
            .to(MemoryRelations.HasSummary, variable="r")
            .node(variable="m")
            .return_("m")
            .execute()
        )
        results = list(query)
        if len(results) == 0:
            return None
        else:
            return results[0]["m"]

    def set_summary(self, summary: str = None):
        # Note here - we can save embeddings for the summary
        # So we can do a similarity search later if we wanted to
        if summary is None:
            summary = "This is a new conversation, the user hasn't spoken yet"
            embedding = None
        else:
            embedding = create_embedding(summary)

        # Upsert the summary node
        curr_summary = self.get_summary()
        if curr_summary is not None:
            curr_summary.content = urllib.parse.quote(summary)
            curr_summary.embedding = embedding
            curr_summary.save(db=db)
            return curr_summary

        summary_node = Summary(
            id=str(uuid()),
            name=self.name + "_summary",
            content=urllib.parse.quote(summary),
            embedding=embedding
        ).save(db=db)

        HasSummary(_start_node_id=self._id, _end_node_id=summary_node._id).save(db=db)
        return summary_node

    def get_system_message(self) -> Message | None:
        query = (
            match()
            .node(node=self, variable="s")
            .to(MemoryRelations.HasSystemMessage, variable="r")
            .node(variable="m")
            .return_("m")
            .execute()
        )
        results = list(query)
        if len(results) == 0:
            return None
        else:
            return results[0]["m"]

    def set_system_message(self, prompt: str):
        print("Prompt", prompt)
        embedding = create_embedding(prompt)

        # Upsert the summary node
        message = self.get_system_message()
        if message is not None:
            print("Existing Message", message)
            message.content = urllib.parse.quote(prompt)
            message.embedding = embedding
            message.save(db=db)
            return message

        message_node = Message(
            id=str(uuid()),
            timestamp=get_timestamp(),
            role=ConversationRole.System.value,
            content=urllib.parse.quote(prompt),
            embedding=embedding
        ).save(db=db)
        print("Existing Message", message)
        HasSystemMessage(_start_node_id=self._id, _end_node_id=message_node._id).save(db=db)
        return message_node

    def get_messages(self):
        query = (
            match()
            .node(node=self, variable="s")
            .to(MemoryRelations.HasMessage, variable="r")
            .node(variable="m")
            .return_("m")
            .order_by(properties=("m.timestamp", Order.ASC))
            .execute()
        )
        results = list(query)
        return results

    def add_message(self, role: ConversationRole, message: str):
        embedding = create_embedding(message)
        messages = self.get_messages()
        print("Messages", messages)
        prev_message = None
        if len(messages) > 0:
            prev_message = messages[-1]

        print("Prev message", prev_message)
        message_node = Message(
            id=str(uuid()),
            timestamp=get_timestamp(),
            role=role,
            content=urllib.parse.quote(message),
            embedding=embedding
        ).save(db=db)
        HasMessage(_start_node_id=self._id, _end_node_id=message_node._id).save(db=db)
        # Link the message to the previous message in the conversation (if that exists)
        if prev_message is not None:
            HasReply(_start_node_id=prev_message["m"]._id, _end_node_id=message_node._id).save(db=db)

        return message_node

    def delete_conversation(self):
        (
            match()
            .node(node=self, variable="s")
            .to(variable="r")
            .node(variable="o")
            .add_custom_cypher("DETACH DELETE s, r, o;")
            .execute()
        )


