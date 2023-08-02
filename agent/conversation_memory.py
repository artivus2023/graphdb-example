from node_types.conversation import Conversation, Summary, Message, HasSummary, HasMessage, HasReply
from utils.enums import ConversationRole
from uuid import uuid4 as uuid
from gqlalchemy import Memgraph

db = Memgraph()

class ConversationMemory():
    def __init__(self, name: str, default_summary: str = None):
        self.conversation_name = name
        self.conversation_summary = default_summary
        self.conversation = self.upsert_conversation()

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