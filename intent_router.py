"""
Intent router: decides whether an incoming message should go to the
psychological support bot or the learning/curriculum bot.

Usage (standalone test):
    python router.py "my daughter is anxious about her exams"
    python router.py "explain photosynthesis for 8th grade"
"""

import json
import os
import sys

import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = "gpt-oss:120b-cloud"

VALID_INTENTS = ["psychological", "learning"]

CLASSIFIER_SYSTEM_PROMPT = """You are a router that decides which assistant should handle a parent's message.

There are two assistants:
- "psychological": handles emotional/behavioral/mental-health topics — a child's
  anxiety, stress, motivation, family conflict, parenting concerns, social
  difficulties, discipline, self-esteem, etc.
- "learning": handles questions about school curriculum content — specific
  subjects (math, science, languages, history, etc.), homework help, what's
  covered in a grade/chapter, explaining a concept from class.

Some messages are ambiguous (e.g. "my son hates math") — use your best judgment
based on what the parent most likely needs help with right now. If a message
mentions both an emotional angle and a subject, prioritize "psychological" if
the emotional/behavioral concern is the main point, otherwise "learning".

Return ONLY a JSON object, no other text, in this exact form:
{"intent": "psychological" or "learning", "confidence": "high" or "low"}
"""


def classify_intent(message: str) -> dict:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]

    try:
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  WARN: router returned non-JSON, defaulting to psychological. Raw: {raw!r}")
        result = {"intent": "psychological", "confidence": "low"}

    if result.get("intent") not in VALID_INTENTS:
        result["intent"] = "psychological"  # safe default: emotional topics need care either way

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python router.py "message to classify"')
    message = sys.argv[1]
    result = classify_intent(message)
    print(f"Message: {message}")
    print(f"-> intent={result['intent']}  confidence={result.get('confidence')}")