import gradio as gr
from chatbot import chatbot_chat

def gradio_Frontend():
  gr.ChatInterface(
    chatbot_chat,
    chatbot=gr.Chatbot(height=300,placeholder="<strong>👋 Hello! Welcome to our chat! I’m an AI-powered assistant here to help you with your questions and provide information.</strong><br> Feel free to ask me anything, and I'll do my best to assist you!"),
    textbox=gr.Textbox(placeholder="Send me queries", container=False, scale=7),
    title="Company Bot",
    description="Ask Company Bot any questions about Roadelabs",
    theme="soft",
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()


