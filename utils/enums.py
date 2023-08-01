from enum import Enum

# Enumerations for the Simulacra graph database node types
class SimulacraNodeType(str, Enum):
    Semantic = 'Semantic'
    WorldState = 'WorldState'
    Conversation = 'Conversation'
    Episodic = 'Episodic'
    Message = 'Message'
    Summary = 'Summary'
    
# Enumerations for the Simulacra graph database relationship types
class MemoryRelations(str, Enum):
    HasMessage = 'HAS_MESSAGE'
    HasSystemMessage = 'HAS_SYSTEM_MESSAGE'
    HasSummary = 'HAS_SUMMARY'
    HasReply = 'HAS_REPLY'
    KnowledgeSource = 'KNOWLEDGE_SOURCE'
    Predicate = 'PREDICATE'
    SemanticMeaning = 'SEMANTIC_MEANING'
    WorldStateAction = 'WORLD_STATE_ACTION'
    
    
# Valid conversation roles
class ConversationRole(str, Enum):
    System = 'system'
    User = 'user'
    Assistant = 'assistant'
    Function = 'function'