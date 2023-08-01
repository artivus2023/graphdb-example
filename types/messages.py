from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from utils.enums import ConversationRole

class BaseMessage(BaseModel, ABC):
    @property
    @abstractmethod
    def role(self) -> ConversationRole:
        pass

    @property
    @abstractmethod
    def content(self) -> ConversationRole:
        pass
    
from utils.enums import ConversationRole

class AIMessage(BaseMessage):
    @property
    def role(self) -> ConversationRole:
        return ConversationRole.AGENT

    @property
    def content(self) -> str:
        return self._content

class UserMessage(BaseMessage):
    @property
    def role(self) -> ConversationRole:
        return ConversationRole.USER

    @property
    def content(self) -> str:
        return self._content

class SystemMessage(BaseMessage):
    @property
    def role(self) -> ConversationRole:
        return ConversationRole.SYSTEM

    @property
    def content(self) -> str:
        return self._content

class FunctionMessage(BaseMessage):
    @property
    def role(self) -> ConversationRole:
        return ConversationRole.FUNCTION

    @property
    def content(self) -> str:
        return self._content