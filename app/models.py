from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    file_path: str
