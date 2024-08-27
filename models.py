from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
import openai
import os
from dotenv import load_dotenv
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

load_dotenv()
openai.api_key = os.getenv("API_KEY")

def load_t5_model_and_tokenizer(model_name):
    
    # Load the T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)  # New behavior
    # Load the T5 model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def get_transformer_embedding_function():
    embeddings = SentenceTransformerEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True})
    return embeddings

def get_openai_embedding_function():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=api_key)
    
    return embeddings

def generate_gpt_response(model,prompt):
    if(model=="gpt-3.5-turbo"):
        response=openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[  {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    elif(model=="gpt-4o"):
        response=openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[  {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response

def generate_t5_response(query_text,context_text):
    # Load the T5 model and tokenizer (if not already loaded)
    model_name="t5-small"    
    tokenizer, model = load_t5_model_and_tokenizer(model_name)
    # Optional: Add special tokens (if needed) and resize embeddings
    special_tokens = {'additional_special_tokens': ['<custom_token>']}
    tokenizer.add_special_tokens(special_tokens)
    # Resize model embeddings to match the tokenizer's vocabulary
    model.resize_token_embeddings(len(tokenizer))
    # Prepare the input for T5
    input_text = f"question: {query_text} context: {context_text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate answer using T5
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=50)  # Set to the desired token length

    response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response_text
    


