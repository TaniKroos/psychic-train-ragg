from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from io import BytesIO

from app.document_processor import extract_text_from_pdf_file, chunk_text
from app.cohere_client import CohereClient
from app.embedding_store import EmbeddingStore

app = FastAPI()
cohere_client = CohereClient()
store = EmbeddingStore()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), username: str = Form(...)):
    contents = await file.read()
    text = extract_text_from_pdf_file(BytesIO(contents))
    chunks = chunk_text(text)
    embeddings = cohere_client.get_embeddings(chunks)
    store.add_embeddings(embeddings, chunks, username)
    return {"status": "success", "chunks_indexed": len(chunks)}


class QueryRequest(BaseModel):
    query: str
    username: str


@app.post("/query")
def query_rag(request: QueryRequest):
    query_embedding = cohere_client.get_embeddings([request.query])[0]
    relevant_chunks = store.search(query_embedding, username=request.username, k=5)

    if not relevant_chunks:
        return {"answer": "No relevant context found for this user."}

    context = "\n\n".join(relevant_chunks)
    answer = cohere_client.generate_response(request.query, context)

    return {
        "query": request.query,
        "answer": answer,
        "used_chunks": relevant_chunks
    }
