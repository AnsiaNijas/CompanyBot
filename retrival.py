from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function


CHROMA_PATH="chroma_data"

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    return (results)