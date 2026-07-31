"""
Step 4: Learning bot query pipeline (standalone test, not yet wired to the API).

Given a parent's question:
  1. Classify grade + subject from the question text (LLM call).
  2. Embed the question (bge-m3, same model used for the chunks).
  3. Search Qdrant, filtered by the classified grade/subject.
  4. Generate a grounded answer citing grade/subject/chapter.

Requires:
    ollama pull bge-m3
    ollama pull gpt-oss:120b-cloud   (or whatever chat model you're using)
    pip install qdrant-client requests python-dotenv

Usage:
    python query_learning_bot.py "How do I explain photosynthesis to my 8th grader?"
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "gpt-oss:120b-cloud"
COLLECTION_NAME = "school_materials"
TOP_K = 5

VALID_GRADES = ["7th", "8th", "9th"]
VALID_SUBJECTS = [
    "history", "islamic_education", "christian_education", "visual_arts",
    "music", "national_education", "math", "physics_chemistry", "english",
    "russian", "arabic", "french", "ict", "geography", "biology_earth_science",
]


def ollama_chat(messages: list[dict], model: str = CHAT_MODEL) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def classify_question(question: str) -> dict:
    system = f"""You classify parents' questions about school material into a grade and subject.

Valid grades: {VALID_GRADES + ["unknown"]}
Valid subjects: {VALID_SUBJECTS + ["unknown"]}

Return ONLY a JSON object, no other text, in this exact form:
{{"grade": "...", "subject": "..."}}

Use "unknown" for either field if the question doesn't make it clear.
Do not guess wildly — only classify grade/subject if reasonably confident."""

    raw = ollama_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ])

    try:
        
        cleaned = raw.strip().strip("`").replace("json\n", "", 1) if raw.strip().startswith("```") else raw
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  WARN: classifier returned non-JSON, defaulting to unknown. Raw: {raw!r}")
        result = {"grade": "unknown", "subject": "unknown"}

    if result.get("grade") not in VALID_GRADES:
        result["grade"] = "unknown"
    if result.get("subject") not in VALID_SUBJECTS:
        result["subject"] = "unknown"

    return result


def search_qdrant(client: QdrantClient, question: str, grade: str, subject: str, top_k: int = TOP_K):
    vector = get_embedding(question)

    conditions = []
    if grade != "unknown":
        conditions.append(FieldCondition(key="grade", match=MatchValue(value=grade)))
    if subject != "unknown":
        conditions.append(FieldCondition(key="subject", match=MatchValue(value=subject)))

    query_filter = Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return response.points


def build_contextual_query(message: str, memory: list) -> str:
    if not memory:
        return message
    last_user_turns = [m["content"] for m in memory if m["role"] == "user"][-2:]
    if not last_user_turns:
        return message
    return " ".join(last_user_turns) + " " + message


def generate_answer_with_memory(question: str, results, memory: list) -> str:
    if not results:
        return "I couldn't find anything relevant in the curriculum material for this question."

    context_blocks = []
    for r in results:
        p = r.payload
        label = f"[Grade {p.get('grade')} | {p.get('subject')} | {p.get('chapter_title') or 'N/A'}]"
        context_blocks.append(f"{label}\n{p.get('text', '')}")
    context = "\n\n---\n\n".join(context_blocks)

    system = """You are a helpful assistant for parents, answering questions about their
child's school curriculum. Use ONLY the provided context to answer. If the context
doesn't fully answer the question, say so honestly rather than inventing facts.
Use the conversation history to understand follow-up questions and maintain
continuity, but always ground the actual answer in the provided context.
At the end, briefly note which grade/subject/chapter the answer came from."""

    messages = [{"role": "system", "content": system}]
    messages.extend(memory)
    messages.append({
        "role": "user",
        "content": f"Context from the curriculum:\n\n{context}\n\n---\n\nParent's question: {question}",
    })

    return ollama_chat(messages)


def generate_answer(question: str, results) -> str:
    if not results:
        return "I couldn't find anything relevant in the curriculum material for this question."

    context_blocks = []
    for r in results:
        p = r.payload
        label = f"[Grade {p.get('grade')} | {p.get('subject')} | {p.get('chapter_title') or 'N/A'}]"
        context_blocks.append(f"{label}\n{p.get('text', '')}")

    context = "\n\n---\n\n".join(context_blocks)

    system = """You are a helpful assistant for parents, answering questions about their
child's school curriculum. Use ONLY the provided context to answer. If the context
doesn't fully answer the question, say so honestly rather than inventing facts.
At the end, briefly note which grade/subject/chapter the answer came from."""

    user_message = f"Context from the curriculum:\n\n{context}\n\n---\n\nParent's question: {question}"

    return ollama_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ])


def main(question: str):
    qdrant_url = os.environ.get("QDRANT_URL")
    if not qdrant_url:
        sys.exit("Missing QDRANT_URL o .env file.")

    print(f"Question: {question}\n")

    print("Classifying...")
    classification = classify_question(question)
    print(f"  -> grade={classification['grade']}, subject={classification['subject']}\n")

    client = QdrantClient(url=qdrant_url, timeout=60)

    print("Searching Qdrant...")
    results = search_qdrant(client, question, classification["grade"], classification["subject"])
    print(f"  -> found {len(results)} result(s)\n")

    for r in results:
        p = r.payload
        print(f"  score={r.score:.3f}  grade={p.get('grade')}  subject={p.get('subject')}  chapter={p.get('chapter_title')}")

    print("\nGenerating answer...\n")
    answer = generate_answer(question, results)
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python query_learning_bot.py "your question here"')
    main(sys.argv[1])