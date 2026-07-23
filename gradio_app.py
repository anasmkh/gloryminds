"""
Gradio chat interface: routes each new conversation to either the
psychological support bot or the learning/curriculum bot, and shows
the routing decision to the user before the actual answer.

Requires:
    pip install gradio requests python-dotenv qdrant-client

Run:
    python gradio_app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import os

import gradio as gr
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from intent_router import classify_intent
from query_learning_bot import classify_question, search_qdrant, generate_answer

load_dotenv()

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "gpt-oss:120b-cloud"

# TODO: replace with your real psychological-bot system prompt
# (e.g. load from prompt.txt, same as your original chatbot script)
with open("prompt.txt", "r") as f:
    system_prompt = f.read()
PSYCH_SYSTEM_PROMPT = system_prompt

qdrant_client = None
if os.environ.get("QDRANT_URL") :
    qdrant_client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        timeout=60,
    )


def ollama_chat(messages: list[dict]) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": CHAT_MODEL, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def psychological_response(message: str, gradio_history: list) -> str:
    """gradio_history is a list of {"role": ..., "content": ...} dicts
    (Gradio's 'messages' format) from prior turns in this conversation."""
    messages = [{"role": "system", "content": PSYCH_SYSTEM_PROMPT}]
    messages.extend(gradio_history)
    messages.append({"role": "user", "content": message})
    return ollama_chat(messages)


def learning_response(message: str) -> str:
    if qdrant_client is None:
        return "Learning bot isn't available: QDRANT_URL/QDRANT_API_KEY are not set in .env."

    classification = classify_question(message)
    results = search_qdrant(qdrant_client, message, classification["grade"], classification["subject"])
    answer = generate_answer(message, results)
    return answer


def respond(message: str, gradio_history: list):
    routing = classify_intent(message)
    intent = routing["intent"]
    confidence = routing.get("confidence", "unknown")

    if intent == "psychological":
        label = f"** سؤالك متعلق بالدعم النفسي لذا سأقدم النصح بما يتلائم مع العادات والتقاليد في مجتمعنا**"
        answer = psychological_response(message, gradio_history)
    else:
        label = f"** سؤالك متعلق بالمنهج العلمي لذا سأقوم بالبحث حسب المستوى الدراسي لابنك ** "
        answer = learning_response(message)

    return f"{label}\n\n---\n\n{answer}"


demo = gr.ChatInterface(
    fn=respond,
    title="Family Support & Learning Assistant",
    description="Ask about your child's schoolwork, or share a parenting/emotional concern — "
                 "you'll be routed to the right assistant automatically.",
    examples=[
        "My daughter is anxious about her upcoming exams",
        "Can you explain the water cycle for 7th grade?",
        "My son refuses to do his math homework and we fight about it every night",
    ],
)

if __name__ == "__main__":
    demo.launch()