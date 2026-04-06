from pydantic import BaseModel, Field
from typing import List


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    max_new_tokens: int = Field(default=100, ge=10, le=500)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)


class ChatResponse(BaseModel):
    reply: str
    turn_count: int
