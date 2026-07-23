from typing import List

import os
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from llama_index.core import Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient

import models
import schemas
from auth import create_access_token, get_current_user, hash_password, verify_password
from database import Base, engine, get_db
from intent_router import classify_intent
from query_learning_bot import (
    classify_question,
    search_qdrant,
    build_contextual_query,
    generate_answer_with_memory,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)  # creates users/chats/messages tables

app = FastAPI(title="Chatbot API")

Settings.llm = Ollama(
    model="gpt-oss:120b-cloud",
    request_timeout=120.0,
    base_url="http://localhost:11434",
)

with open("prompt.txt", "r") as f:
    system_prompt = f.read()

qdrant_client = None
if os.environ.get("QDRANT_URL") and os.environ.get("QDRANT_API_KEY"):
    qdrant_client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60,
    )

# --- Memory buffer settings ------------------------------------------------
# Keep only the last N turns of DB-stored history when building context for
# either bot, so the prompt sent to the LLM stays bounded on long chats.
MAX_TURNS = 8
MAX_MESSAGES = MAX_TURNS * 2


def build_memory_messages(db_messages: List[models.Message]) -> List[dict]:
    """Bounded memory buffer built from DB history: last MAX_MESSAGES only."""
    trimmed = db_messages[-MAX_MESSAGES:] if len(db_messages) > MAX_MESSAGES else db_messages
    return [{"role": m.role, "content": m.content} for m in trimmed]


def build_chat_engine(db_messages: List[models.Message]) -> SimpleChatEngine:
    """Rebuild the psychological bot's chat engine, seeded with bounded
    memory from stored history (llama-index's own token_limit already
    bounds this further at the token level)."""
    memory_messages = build_memory_messages(db_messages)
    history = [
        ChatMessage(
            role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
            content=m["content"],
        )
        for m in memory_messages
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096, chat_history=history)
    return SimpleChatEngine.from_defaults(
        memory=memory,
        system_prompt=system_prompt,
        llm=Settings.llm,
    )


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
async def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    user = models.User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username or email already registered")
    db.refresh(user)
    return user


@app.post("/login", response_model=schemas.Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/me", response_model=schemas.UserOut)
async def me(current_user: models.User = Depends(get_current_user)):
    return current_user


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=schemas.ChatDetailOut)
async def chat(
    request: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if request.chat_id:
        chat_obj = (
            db.query(models.Chat)
            .filter(models.Chat.id == request.chat_id, models.Chat.user_id == current_user.id)
            .first()
        )
        if not chat_obj:
            raise HTTPException(status_code=404, detail="Chat not found")
        bot_type = chat_obj.bot_type or "psychological"  # fallback for old rows predating this column
    else:
        # New chat: classify intent once, then lock it in for this chat's lifetime.
        routing = classify_intent(request.message)
        bot_type = routing["intent"]
        chat_obj = models.Chat(
            user_id=current_user.id,
            title=request.message[:50],
            bot_type=bot_type,
        )
        db.add(chat_obj)
        db.commit()
        db.refresh(chat_obj)

    memory = build_memory_messages(chat_obj.messages)

    if bot_type == "psychological":
        engine_instance = build_chat_engine(chat_obj.messages)
        response = engine_instance.chat(request.message)
        answer_text = str(response)
    else:  # "learning"
        if qdrant_client is None:
            answer_text = "The learning assistant isn't available right now (Qdrant is not configured)."
        else:
            contextual_query = build_contextual_query(request.message, memory)
            classification = classify_question(contextual_query)
            results = search_qdrant(
                qdrant_client, contextual_query, classification["grade"], classification["subject"]
            )
            answer_text = generate_answer_with_memory(request.message, results, memory)

    db.add_all(
        [
            models.Message(chat_id=chat_obj.id, role="user", content=request.message),
            models.Message(chat_id=chat_obj.id, role="assistant", content=answer_text),
        ]
    )
    db.commit()
    db.refresh(chat_obj)

    return chat_obj


@app.get("/chats", response_model=List[schemas.ChatOut])
async def list_chats(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return (
        db.query(models.Chat)
        .filter(models.Chat.user_id == current_user.id)
        .order_by(models.Chat.created_at.desc())
        .all()
    )


@app.get("/chats/{chat_id}", response_model=schemas.ChatDetailOut)
async def get_chat(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    chat_obj = (
        db.query(models.Chat)
        .filter(models.Chat.id == chat_id, models.Chat.user_id == current_user.id)
        .first()
    )
    if not chat_obj:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat_obj


@app.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    chat_obj = (
        db.query(models.Chat)
        .filter(models.Chat.id == chat_id, models.Chat.user_id == current_user.id)
        .first()
    )
    if not chat_obj:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(chat_obj)
    db.commit()


@app.get("/")
async def root():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)