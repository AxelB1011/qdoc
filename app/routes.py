from app import app
from fastapi import HTTPException
from .models import QueryRequest, IngestRequest
from .services import ingest_data, query_data

@app.post("/ingest")
def ingest_endpoint(request: IngestRequest):
    return ingest_data(request)

@app.post("/query")
def query_endpoint(request: QueryRequest):
    return query_data(request)
