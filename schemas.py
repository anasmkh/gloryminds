from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatOut(BaseModel):
    id: str
    title: str
    bot_type: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatDetailOut(ChatOut):
    messages: List[MessageOut] = []


class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None  # omit to start a new chat