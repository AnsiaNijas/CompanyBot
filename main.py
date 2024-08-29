from Database import load_pdf_documents, spilt_pdf_documents, insert_into_chroma
from chatbot import chatbot_chat
from Database import query_rag
from frontend import gradio_Frontend


if __name__ == "__main__":
            
    # Load and process documents
    documents = load_pdf_documents()
    chunks = spilt_pdf_documents(documents)
    insert_into_chroma(chunks)
    gradio_Frontend()
