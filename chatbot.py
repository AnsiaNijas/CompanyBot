from langchain_community.vectorstores import Chroma
import openai
from models import  get_embedding_function, generate_gpt_response,generate_t5_response
import argparse
from langchain.prompts import ChatPromptTemplate
from Database import query_rag
import os 
from dotenv import load_dotenv
import re  # Importing regular expressions module for pattern matching
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

load_dotenv()

openai.api_key = os.getenv("API_KEY")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Function to check if the question is out of scope (e.g., asking for personal details)
def is_question_out_of_scope(question):
    # List of keywords related to personal information
    personal_info_keywords = [
        "personal", "address", "phone", "email", 
        "social security", "account", "birth date", "age", 
        "gender", "location", "bank account", "credit card", "ID", 
        "login", "password", "SSN", "birthday"
    ]

    # Check for patterns in the question
    pattern = re.compile(r'\b(?:' + '|'.join(personal_info_keywords) + r')\b', re.IGNORECASE)
    
    # Return True if any personal info keywords are found in the question
    return bool(pattern.search(question))

# Function to check for greeting messages
def is_greeting_message(message):
    greeting_keywords = ["hi ", "hello", "hey", "howdy", "greetings", "what's up", "welcome"]
    return any(greet in message.lower() for greet in greeting_keywords)

def chatbot_chat(query_text, history):
    # Check for a greeting message
    if is_greeting_message(query_text):
        return "Hello! How can I assist you today?"
    # Check if the question is out of scope
    if is_question_out_of_scope(query_text):
        return "I'm sorry, but I can't provide personal details or answer questions that require sensitive information. I can only respond to questions related to our company."

    # Proceed with querying RAG for relevant information
    results = query_rag(query_text)

    # Check if any results were found
    if not results:
        return "I'm sorry, but I couldn't find any relevant information on that topic. Please ask a different question."

    # Process the results if found
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Sending appropriate prompt to the GPT model
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Get the response from the GPT-4o function
    response = generate_gpt_response("gpt-4o",prompt)
    # Ensure a valid response is returned
    if response and hasattr(response.choices[0], "message"):
        response_text = response.choices[0].message["content"]
         # Adjust the response message if the content indicates the question is out of scope
        if "does not" in response_text.lower():
                return "I'm sorry, but this question requires information that I can't provide. Please ask questions related to our company or our services."
        
    # Retrieve sources from the results
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # Format the final response
    formatted_response = f"Response: {response_text}\nSources: {sources if sources else 'No sources available.'}"
    
    return formatted_response


def chatbot_chat1(query_text,model):
   
    results=query_rag(query_text)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    if (model == "gpt-4o") or (model == "gpt-3.5-turbo"):
        response = generate_gpt_response(model,prompt)
        response_text = response.choices[0].message["content"]
    
    elif(model == "t5"):
        response_text=generate_t5_response(query_text,context_text)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response


   

