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
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

#  Pinecone vector database
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")


from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key, environment="us-east-1")

index_name = '3rdasset'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

#  Embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"])

#  Request Entry
class Entry(BaseModel):
    query_prompt: str
    file_link: str
    description: str
    category: str

class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    sort_by_date: Optional[bool] = False

def generate_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

#  Insert endpoint
@app.post("/insert")
async def insert_entry(entry: Entry):
    vector = generate_embedding(entry.description)

    unique_id = str(uuid.uuid4())
    metadata = {
        "unique_id" : unique_id,
        "category" : entry.category,  
        "file_link" : entry.file_link,  
        "query_prompt" : entry.query_prompt,  
        "description" : entry.description
    }
    
    vector_tuple = (unique_id, vector, metadata)

    index.upsert(
        vectors=[vector_tuple],
        namespace="ns1"
    )
    index.describe_index_stats()

    return {"message": "Entry added successfully"}

#  search endpoint
@app.get("/search")
async def search_similar(query: str):
    
    query_vector = generate_embedding(query)
    
    result = index.query(
        namespace="ns1",
        vector=query_vector,
        top_k=4,
        include_values=True,
        include_metadata=True
    )
    # print(type(result))
    # print(result)
    # metadata = result['matches'][0]['metadata']

    # return {"file_link": metadata["file_link"],
    #         "description": metadata["description"]}

    metadata_list = []

    for match in result['matches']:
        metadata_list.append(match['metadata'])
    return metadata_list

# get all 3dobject
@app.get("/getallitems")
async def get_all_items():
    results=[]
    for id in index.list(namespace='ns1'):
        for i in id:
            # print(i)
            result = index.query(
                namespace="ns1",
                id=str(i),
                top_k=1,
                include_values=False,
                include_metadata=True
            )
            metadata_list = []
            for match in result['matches']:
                metadata_list.append(match['metadata'])
            # print(metadata_list)
            results.extend(metadata_list)
    return results


# ......filterSearch......
@app.post("/filtersearch")
async def search_similar(search_request: SearchRequest):
    # Generate vector embedding for the search query
    query_vector = generate_embedding(search_request.query)
    
    # Create filter for category and subcategory if provided
    filter_query = {}
    if search_request.category:
        filter_query["category"] = {"$eq": search_request.category}
    if search_request.subcategory:
        filter_query["subcategory"] = {"$eq": search_request.subcategory}

    # Query Pinecone for similar vectors with filtering and top_k=4
    result = index.query(
        namespace="ns1",
        vector=query_vector,
        top_k=4,
        include_values=False,
        include_metadata=True,
        filter=filter_query
    )
    
    # Check if there are any matches
    if not result.matches:
        raise HTTPException(status_code=404, detail="No similar entries found")
    
    # Extract metadata from the matches
    metadata_list = [match['metadata'] for match in result.matches]

    # Sort matches by date_of_creation if requested
    if search_request.sort_by_date:
        metadata_list = sorted(metadata_list, key=lambda x: x.get('date_of_creation', ''), reverse=True)
    
    # Return the sorted metadata list
    return metadata_list

# @app.post("/filtersearch")
# async def search_similar(search_request: SearchRequest):
    # query_vector = generate_embedding(search_request.query)
    # Create filter for category and subcategory if provided
    # filter = {}
    # if search_request.category:
    #     filter["category"] = {"$eq": search_request.category}
    # if search_request.subcategory:
    #     filter["subcategory"] = {"$eq": search_request.subcategory}

    # Query Pinecone for similar vectors
    # result = index.query(
    #     namespace="ns1",
    #     vector=query_vector,
    #     top_k=4,
    #     include_values=True,
    #     filter=filter
    # )
    
    # if not result or not result.matches:
    #     raise HTTPException(status_code=404, detail="No similar entries found")
    
    # metadata_list = []

    # for match in result['matches']:
    #     metadata_list.append(match['metadata'])
    # return metadata_list

    # Extract matches and sort by date_of_creation if requested
    # matches = [match['metadata'] for match in result['matches']]
    # if search_request.sort_by_date:
    #     matches = sorted(matches, key=lambda x: x['date_of_creation'], reverse=True)
    # return {"matches": matches}

    # metadata_list = []

    # for match in result['matches']:
    #     metadata_list.append(match['metadata'])
    # return metadata_list
# ......
    
# findbyid_and_return_data
@app.get("/findbyid/{id}")
async def findbyid_and_return_data(id: str):
    result = index.query(
        namespace="ns1",
        id=id,
        top_k=1,
        include_values=False,
        include_metadata=True
    )

    metadata_list = []

    for match in result['matches']:
        metadata_list.append(match['metadata'])
    return metadata_list

#  start app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
