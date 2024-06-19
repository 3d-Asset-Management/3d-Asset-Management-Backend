import pinecone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
import uvicorn
import os 
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from langchain_pinecone import PineconeVectorStore

# Ensure the Pinecone API key is set
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key, environment="us-east-1")

# Create or connect to Pinecone index
index_name = '3rdasset'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Load the model and tokenizer for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# FastAPI app initialization
app = FastAPI()

# Define the request model
class Entry(BaseModel):
    query_prompt: str
    file_link: str
    description: str
    category: str

def generate_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

@app.post("/insert")
async def insert_entry(entry: Entry):
    # Generate a vector representation of the entry
    # print(entry.description)
    # print(entry.category)
    # print(entry.file_link)
    # print(entry.description)
    vector = generate_embedding(entry.description)

    unique_id = str(uuid.uuid4())
    metadata = {
    "category": entry.category,  # Ensure category is converted to string
    "file_link": entry.file_link,  # Ensure file_link is converted to string
    "query_prompt": entry.query_prompt,  # Ensure query_prompt is converted to string
    "description": entry.description
    }
    # index.upsert(
    #     vectors=[
    #         {"id": f"{unique_id}", "values": vector}
    #     ],
    #     metadata=metadata,
    #     namespace="ns1"
    # )
    vector_tuple = (unique_id, vector, metadata)

    index.upsert(
        vectors=[vector_tuple],
        namespace="ns1"
    )
    index.describe_index_stats()

    return {"message": "Entry added successfully"}

@app.get("/search")
async def search_similar(query: str):
    # Generate a vector for the query
    query_vector = generate_embedding(query)
    
    # Query Pinecone for similar vectors
    result = index.query(
        namespace="ns1",
        vector=query_vector,
        top_k=1,
        include_values=True,
        include_metadata=True
    )
    print(type(result))
    print(result)
    metadata = result['matches'][0]['metadata']

    # Extract metadata from the results
    # matches = []
    # for match in result[0]:
    #     metadata = match['values'][2]  # Access metadata correctly
    #     matches.append(metadata)

    # return matches
    
    # if not result:
    #     raise HTTPException(status_code=404, detail="No similar entries found")

    # # Return the matched entries
    # matches = [match[1]['metadata'] for match in result[0]]
    return {"file_link": metadata["file_link"],
            "description": metadata["description"]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
