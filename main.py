from fastapi import FastAPI, UploadFile, File, HTTPException
from Database import load_documents, split_documents, add_to_chroma, check_clear_database
from chatbot import chatbot_chat
from retrival import query_rag

app = FastAPI()
CHROMA_PATH="chroma_data"
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    #check_clear_database()
    documents=load_documents()
    chunks=split_documents(documents)
    add_to_chroma(chunks)
    return {"Loaded":{len(documents)},"chunks":chunks,"filename": file.filename}

@app.get("/")
def read_root():
    formatted_response=chatbot_chat()
    return {"Response": formatted_response}