"""
Gradio chat interface that talks to the real FastAPI backend (/chat) instead
of calling the bot logic directly. Shows which bot handled each message
(psychological vs learning) and reveals the answer progressively.

Note on "streaming": the FastAPI /chat endpoint currently returns the full
answer in one response (it needs the complete text before it can save it to
the DB), not token-by-token. This simulates a stream by revealing the
already-complete answer word by word — same UX feel, no backend rewrite
needed. For genuine token-level streaming end to end, main.py's /chat
endpoint would need to become a real streaming endpoint (Server-Sent Events)
that streams from Ollama as it generates — a separate, bigger change.

Requires:
    pip install gradio requests

Run (with your FastAPI server already running on port 8000):
    python gradio_app.py
"""

import time

import gradio as gr
import requests

API_BASE_URL = "http://localhost:8000"  # assumes your Docker container publishes port 8000 -> 8000
STREAM_DELAY = 0.02  # seconds between words; tune for faster/slower reveal

INTENT_LABELS = {
    "psychological": "🧭 **تم اختيار: المساعد النفسي** _(Psychological Assistant)_",
    "learning": "🧭 **تم اختيار: المساعد التعليمي** _(Learning Assistant)_",
}

# Holds the current chat_id so follow-up messages continue the same
# conversation server-side (bot_type stays locked, memory stays bounded).
# NOTE: this is a single shared variable, fine for one person testing
# locally. It is NOT per-browser-session-safe for multiple simultaneous
# users — each user would share/clobber the same chat_id. Use gr.State
# instead if this ever needs to support multiple concurrent testers.
_session = {"chat_id": None}


def extract_text(content) -> str:
    """Gradio 6.x sometimes represents message content as a list of parts
    (e.g. [{"text": "hi", "type": "text"}]) rather than a plain string, even
    for plain text input. Normalize either shape down to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return str(content)


def call_chat_api(message: str) -> dict:
    payload = {"message": message}
    if _session["chat_id"]:
        payload["chat_id"] = _session["chat_id"]

    print("SENDING PAYLOAD:", payload)  # temporary debug line

    resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=120)

    if not resp.ok:
        print("ERROR STATUS:", resp.status_code)
        print("ERROR BODY:", resp.text)  # temporary debug lines

    resp.raise_for_status()
    data = resp.json()
    print("RECEIVED bot_type:", data.get("bot_type"), "| chat id:", data.get("id"))  # temporary debug line
    _session["chat_id"] = data["id"]
    return data


def user_submit(message: str, history: list):
    if not message.strip():
        return history, message
    history = history + [{"role": "user", "content": message}]
    return history, ""


def bot_respond(history: list):
    if not history or history[-1]["role"] != "user":
        return

    user_message = extract_text(history[-1]["content"])

    try:
        data = call_chat_api(user_message)
    except requests.exceptions.RequestException as e:
        history.append({"role": "assistant", "content": f"⚠️ Could not reach the API: {e}"})
        yield history
        return

    bot_type = data.get("bot_type")
    label = INTENT_LABELS.get(bot_type, f"🧭 **Routed to: {bot_type}**")

    assistant_messages = [m for m in data["messages"] if m["role"] == "assistant"]
    answer = assistant_messages[-1]["content"] if assistant_messages else "(no response)"

    header = f"{label}\n\n---\n\n"

    # Show the routing decision immediately, on its own, before any of the
    # answer appears — this is the "tell the user which bot first" behavior.
    history.append({"role": "assistant", "content": header})
    yield history
    time.sleep(0.4)  # brief pause so the routing label is clearly seen first

    # Then stream the answer word by word underneath the (already-shown) label
    words = answer.split(" ")
    partial = ""
    for i, word in enumerate(words):
        partial += (" " if i > 0 else "") + word
        history[-1]["content"] = header + partial
        yield history
        time.sleep(STREAM_DELAY)


def new_conversation():
    _session["chat_id"] = None
    return []


with gr.Blocks(title="GloryMinds Bot") as demo:
    gr.Markdown(
        "# Family Support & Learning Assistant\n"
        "Ask about your child's schoolwork, or share a parenting/emotional concern — "
        f"you'll be routed to the right assistant automatically. _(Live API: {API_BASE_URL})_"
    )
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type your message and press Enter...", show_label=False)
    new_chat_btn = gr.Button("🔄 New Conversation")

    msg.submit(user_submit, [msg, chatbot], [chatbot, msg], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    new_chat_btn.click(new_conversation, None, chatbot, queue=False)

demo.queue()

if __name__ == "__main__":
    demo.launch()