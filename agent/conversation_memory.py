from node_types.conversation import Conversation, Summary, Message, HasSummary, HasMessage, HasReply
from utils.enums import ConversationRole, SimulacraNodeType
from uuid import uuid4 as uuid
from gqlalchemy import Memgraph, match

db = Memgraph()

class ConversationMemory():
    def __init__(self, name: str, default_summary: str = None):
        self.conversation_name = name
        self.conversation_summary = default_summary
        self.conversation = self.upsert_conversation()

    # Create a converstation graph if one doesn't already exist
    def upsert_conversation(self):
        try:
            conversation = Conversation(name=self.conversation_name).load(db=db)
        except Exception as e:
            conversation = Conversation(
                name=self.conversation_name,
                description="A conversation between a user and an AI assistant",
                id=str(uuid())
            ).save(db=db)

        # Create default summary node
        conversation.set_summary(self.conversation_summary)
        return conversation

    def add_message(self, role: ConversationRole, message: str):
        # Add the message to the conversation
        message_node = self.conversation.add_message(role=role.value, message=message)
        return message_node
    
    
# TODO: Do this properly with indexes etc
def get_all_messages_all_conversations():
    query = (
        match()
        .node(labels=SimulacraNodeType.Conversation, variable="m")
        .return_("m")
        .execute()
    )
    results = list(query)
    if len(results) == 0:
        return None
    else:
        return [result["m"]["message"] for result in results]
        