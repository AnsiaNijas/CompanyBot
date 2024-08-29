from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from models import get_openai_embedding_function
import argparse
import shutil

# Load environment variables
load_dotenv()

# Function to load PDF documents from the specified directory
def load_pdf_documents():
    pdf_loader_instance = PyPDFDirectoryLoader(os.getenv("DATA_PATH"))
    return pdf_loader_instance.load()

def spilt_pdf_documents(documents: list[Document]):
    # Create a text splitter for managing the size of PDF chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

# Function to generate unique IDs for document chunks
def generate_chunk_ids(chunk_list):
    previous_page_id = None
    chunk_index = 0

    for single_chunk in chunk_list:
        source_file = single_chunk.metadata.get("source")
        page_number = single_chunk.metadata.get("page")
        current_identifier = f"{source_file}:{page_number}"

        # Increment index if the current page ID is same as the last one
        if current_identifier == previous_page_id:
            chunk_index += 1
        else:
            chunk_index = 0

        # Generate the chunk ID and update the meta-data
        generated_id = f"{current_identifier}:{chunk_index}"
        previous_page_id = current_identifier
        single_chunk.metadata["id"] = generated_id

    return chunk_list

def insert_into_chroma(chunk_list: list[Document]):
    clear_existing_database()
    # Initialize the existing database connection
    database = Chroma(
        persist_directory=os.getenv("CHROMA_PATH"), embedding_function=get_openai_embedding_function()
    )

    # Assign unique IDs to chunks
    chunks_with_generated_ids = generate_chunk_ids(chunk_list)

    # Add or update the documents in the database
    stored_items = database.get(include=[])  # Default includes IDs
    stored_ids = set(stored_items["ids"])
    print(f"Current documents in the database: {len(stored_ids)}")

    # Gather chunks that are not present in the database
    new_chunks_to_add = []
    for chunk in chunks_with_generated_ids:
        if chunk.metadata["id"] not in stored_ids:
            new_chunks_to_add.append(chunk)

    if new_chunks_to_add:
        print(f"👉 Adding new documents: {len(new_chunks_to_add)}")
        new_document_ids = [chunk.metadata["id"] for chunk in new_chunks_to_add]
        database.add_documents(new_chunks_to_add, ids=new_document_ids)
        database.persist()
    else:
        print("✅ No new documents to add")

def clear_existing_database():
    if os.path.exists(os.getenv("CHROMA_PATH")):
        shutil.rmtree(os.getenv("CHROMA_PATH"))

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_openai_embedding_function()
    db = Chroma(persist_directory=os.getenv("CHROMA_PATH"), embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    return (results)