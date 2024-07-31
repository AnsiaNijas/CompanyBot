from langchain_community.vectorstores import Chroma
import openai
from models import get_embedding_function
import argparse
from langchain.prompts import ChatPromptTemplate
from Database import query_rag
import os 
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("API_KEY")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def chatbot_chat(query_text,history):
   
    results=query_rag(query_text)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[  {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
 )

    response_text = response.choices[0].message["content"]
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response


   

