import gradio as gr
import requests

API_URL = "http://localhost:8000"


def chat(message, history, chat_id):
    payload = {
        "message": message,
    }

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=300,
        )

        response.raise_for_status()
        data = response.json()

        # Save chat id after first message
        chat_id = data["id"]

        answer = data["messages"][-1]["content"]

    except Exception as e:
        answer = f"❌ Error:\n{e}"

    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return "", history, chat_id


def new_chat():
    return [], None


with gr.Blocks(title="AI Assistant") as demo:

    gr.Markdown("# 🤖 AI Assistant")

    chat_id = gr.State(None)

    chatbot = gr.Chatbot(
        label="Conversation",
        height=600,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message...",
            scale=8,
        )

        send = gr.Button("Send", scale=1)

    clear = gr.Button("🆕 New Chat")

    send.click(
        chat,
        inputs=[
            msg,
            chatbot,
            chat_id,
        ],
        outputs=[
            msg,
            chatbot,
            chat_id,
        ],
    )

    msg.submit(
        chat,
        inputs=[
            msg,
            chatbot,
            chat_id,
        ],
        outputs=[
            msg,
            chatbot,
            chat_id,
        ],
    )

    clear.click(
        new_chat,
        outputs=[
            chatbot,
            chat_id,
        ],
    )

demo.launch()