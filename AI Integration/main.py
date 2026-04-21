from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
import os
from load_document import load_documents
from searching.hybrid import hybrid_langchain_retriever
from rag_pipeline import create_chunks, create_vector_store, loaded_vector_store, all_chunks
from typing import List, Annotated

app = FastAPI()

# pydantic class for input
class Question(BaseModel):
    text: str = Field(..., title="The question to ask")
    


# API endpoint to handle document (files: pdf, txt, docx) uploading
@app.post("/upload")
def upload_documents(files: Annotated[List[UploadFile], File(...)]):
    documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
    os.makedirs(documents_dir, exist_ok=True)
    
    uploaded_filenames = []
    
    # save uploaded files in the document directory
    for file in files:
        folder_path = os.path.join(documents_dir, file.filename)
        with open(folder_path, "wb") as f:
            f.write(file.file.read())
        uploaded_filenames.append(file.filename)
    
    # load the uploaded documents
    documents = load_documents(folder_path=documents_dir)
    
    # create chunks of documents
    document_chunks = create_chunks(chunk_size=600, chunk_overlap=150, documents=documents)
    
    # create vector store
    vector_store = create_vector_store(chunks=document_chunks)
    
    return {
        "message": f"{len(uploaded_filenames)} document(s) uploaded successfully.",
        "uploaded_files": uploaded_filenames,
        "Length of loaded document pages": len(documents)
    }


# RAG pipeline logic will go here, including loading documents, creating chunks, creating vector store, and creating retriever.
# API endpopint to handle question answering using the RAG pipeline
@app.post("/ask")
def ask_question(question: Question):
    # get relevent documents from vector store against the question
    relevent_documents = hybrid_langchain_retriever(
        query=question.text,
        lc_documents=all_chunks,
        vectorstore=loaded_vector_store,
        k=5
    )
    
    
    
    
    
    # Placeholder for the actual RAG pipeline logic
    return {
        "response": "This is a placeholder response.",
        "retrieved_documents": relevent_documents
        
        }