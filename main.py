from Database import load_documents, split_documents, add_to_chroma, check_clear_database
from chatbot import chatbot_chat
from Database import query_rag
from frontend import gradio_Frontend


if __name__ == "__main__":
    check_clear_database()
        
    # Load and process documents
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    gradio_Frontend()
