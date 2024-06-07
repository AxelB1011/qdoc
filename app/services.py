# from http.client import HTTPException
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.llama_cpp import LlamaCPP
# from llama_index.readers.file import PyMuPDFReader
# from llama_index.vector_stores.postgres import PGVectorStore
# from llama_index.core.schema import TextNode, NodeWithScore
# from llama_index.core.vector_stores import VectorStoreQuery
# from llama_index.core.query_engine import RetrieverQueryEngine
# import psycopg2
# from .models import QueryRequest, IngestRequest
# import os

# # Initialize components
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# # llm = LlamaCPP(
# #     model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf",
# #     temperature=0.1,
# #     max_new_tokens=256,
# #     context_window=3900,
# #     model_kwargs={"n_gpu_layers": 1},
# #     verbose=True,
# # )

# llm = LlamaCPP(
#     model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf",
#     temperature=0.1,
#     max_new_tokens=256,
#     context_window=3900,
#     model_kwargs={"n_gpu_layers": 1},
#     verbose=True,
# )

# # Database connection
# db_name = os.getenv("DB_NAME", "vector_db")
# host = os.getenv("DB_HOST", "localhost")
# password = os.getenv("DB_PASSWORD", "password")
# port = os.getenv("DB_PORT", "5432")
# user = os.getenv("DB_USER", "user")

# conn = psycopg2.connect(dbname=db_name, host=host, password=password, port=port, user=user)
# conn.autocommit = True

# with conn.cursor() as c:
#     c.execute(f"DROP DATABASE IF EXISTS {db_name}")
#     c.execute(f"CREATE DATABASE {db_name}")

# vector_store = PGVectorStore.from_params(
#     database=db_name,
#     host=host,
#     password=password,
#     port=port,
#     user=user,
#     table_name="llama2_paper",
#     embed_dim=384,
# )

# def ingest_data(request: IngestRequest):
#     loader = PyMuPDFReader()
#     documents = loader.load(file_path=request.file_path)
    
#     text_chunks = [text for doc in documents for text in doc.text.split()]
#     nodes = [TextNode(text=chunk) for chunk in text_chunks]
#     for node in nodes:
#         node_embedding = embed_model.get_text_embedding(node.get_content())
#         node.embedding = node_embedding

#     vector_store.add(nodes)
#     return {"status": "success", "num_documents": len(documents)}

# def query_data(request: QueryRequest):
#     query_embedding = embed_model.get_query_embedding(request.query)
#     vector_store_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2, mode="default")
#     query_result = vector_store.query(vector_store_query)

#     if not query_result.nodes:
#         raise HTTPException(status_code=404, detail="No relevant documents found")

#     response_text = llm(str(query_result.nodes[0].get_content()))
#     return {"response": response_text, "source": query_result.nodes[0].get_content()}

import openai
import os
from http.client import HTTPException
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from .models import QueryRequest, IngestRequest
import psycopg2

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Database connection
db_name = os.getenv("DB_NAME", "vector_db")
host = os.getenv("DB_HOST", "localhost")
password = os.getenv("DB_PASSWORD", "password")
port = os.getenv("DB_PORT", "5432")
user = os.getenv("DB_USER", "user")

conn = psycopg2.connect(dbname=db_name, host=host, password=password, port=port, user=user)
conn.autocommit = True

# with conn.cursor() as c:
#     c.execute(f"DROP DATABASE IF EXISTS {db_name}")
#     c.execute(f"CREATE DATABASE {db_name}")

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="llama2_paper",
    embed_dim=1536,  # Assuming OpenAI's embeddings have this dimension, 3072 for large
)

def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # Adjust model as needed
    )
    return response['data'][0]['embedding']

def ingest_data(request: IngestRequest):
    loader = PyMuPDFReader()
    documents = loader.load(file_path=request.file_path)
    
    text_chunks = [text for doc in documents for text in doc.text.split()]
    nodes = [TextNode(text=chunk) for chunk in text_chunks]
    for node in nodes:
        node_embedding = get_openai_embedding(node.get_content())
        node.embedding = node_embedding

    vector_store.add(nodes)
    return {"status": "success", "num_documents": len(documents)}

def query_data(request: QueryRequest):
    query_embedding = get_openai_embedding(request.query)
    vector_store_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2, mode="default")
    query_result = vector_store.query(vector_store_query)

    if not query_result.nodes:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    response = openai.Completion.create(
        engine="text-davinci-003",  # Adjust model as needed
        prompt=query_result.nodes[0].get_content(),
        max_tokens=150
    )

    response_text = response.choices[0].text.strip()
    return {"response": response_text, "source": query_result.nodes[0].get_content()}
