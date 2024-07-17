from fastapi import FastAPI, UploadFile, File, HTTPException
from Database import load_documents, split_documents, add_to_chroma, check_clear_database

app = FastAPI()
CHROMA_PATH="chroma_data"
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    #check_clear_database()
    documents=load_documents()
    chunks=split_documents(documents)
    add_to_chroma(chunks)
    return {"Loaded":{len(documents)},"chunks":chunks,"filename": file.filename}