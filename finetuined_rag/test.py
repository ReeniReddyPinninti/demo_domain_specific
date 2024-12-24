# from fastapi import FastAPI, HTTPException, UploadFile, Form
# from pydantic import BaseModel
# from typing import List
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# import os

# app = FastAPI()

# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# chunk_size = 256
# chunk_overlap = 25

# def process_and_store_pdf(username: str, pdf_path: str):
#     # Read and process PDF
#     documents = SimpleDirectoryReader(pdf_path).load_data()
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

# @app.post("/embeddings")
# async def generate_embeddings(username: str = Form(...), pdf_file: UploadFile = None):
#     if not pdf_file:
#         raise HTTPException(status_code=400, detail="PDF file is required")

#     try:
#         pdf_path = f"/tmp/{pdf_file.filename}"
#         with open(pdf_path, "wb") as f:
#             f.write(await pdf_file.read())

#         # Generate embeddings and store in MongoDB
#         process_and_store_pdf(username, pdf_path)

#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if os.path.exists(pdf_path):
#             os.remove(pdf_path)

# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# @app.post("/getResponse")
# async def llm_answers(username: str, query: str):
#     try:
#         # Retrieve user embeddings
#         embeddings_data = retrieve_embeddings(username)

#         # Reconstruct the index from stored embeddings
#         index = VectorStoreIndex.from_index_struct(VectorStoreIndex.IndexStruct.from_dict(embeddings_data), embed_model=embed_model)

#         # Setup retriever and query engine
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

#         # Perform query
#         response = query_engine.query(query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         return {"response": response.response, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


# from fastapi import FastAPI, HTTPException, UploadFile, Body
# from pydantic import BaseModel
# from typing import List
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# import os

# app = FastAPI()

# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# chunk_size = 256
# chunk_overlap = 25

# class EmbeddingRequest(BaseModel):
#     username: str
#     pdf_path: str

# def process_and_store_pdf(username: str, pdf_path: str):
#     # Read and process PDF
#     documents = SimpleDirectoryReader(pdf_path).load_data()
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

# @app.post("/embeddings")
# # async def generate_embeddings(request: EmbeddingRequest):
# #     if not request.pdf_path:
# #         raise HTTPException(status_code=400, detail="PDF path is required")

# #     try:
# #         pdf_path = request.pdf_path
# #         # Generate embeddings and store in MongoDB
# #         process_and_store_pdf(request.username, pdf_path)

# #         return {"message": "Embeddings generated and stored successfully"}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
# async def generate_embeddings(request: EmbeddingRequest):
#     print(f"Received PDF path: {request.pdf_path}")  # Add logging to check path

#     if not request.pdf_path:
#         raise HTTPException(status_code=400, detail="PDF path is required")

#     pdf_path = os.path.normpath(request.pdf_path)  # Normalize the path

#     # Check if the path exists and is a file
#     if not os.path.isfile(pdf_path):
#         raise HTTPException(status_code=404, detail=f"Path {pdf_path} is not a file or does not exist")
#     if os.path.isdir(pdf_path):
#         raise HTTPException(status_code=400, detail=f"The path {pdf_path} is a directory, not a file")

#     try:
#         # Process the PDF file
#         process_and_store_pdf(request.username, pdf_path)
#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# @app.post("/getResponse")
# async def llm_answers(username: str, query: str):
#     try:
#         # Retrieve user embeddings
#         embeddings_data = retrieve_embeddings(username)

#         # Reconstruct the index from stored embeddings
#         index = VectorStoreIndex.from_index_struct(VectorStoreIndex.IndexStruct.from_dict(embeddings_data), embed_model=embed_model)

#         # Setup retriever and query engine
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

#         # Perform query
#         response = query_engine.query(query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         return {"response": response.response, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


# from fastapi import FastAPI, HTTPException, UploadFile, File
# from pydantic import BaseModel
# from typing import List
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# import io
# from PyPDF2 import PdfReader
# import os

# app = FastAPI()

# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# class EmbeddingRequest(BaseModel):
#     username: str

# def process_and_store_pdf(username: str, pdf_file: UploadFile):
#     # Read and process PDF
#     pdf_data = pdf_file.file.read()
#     pdf_reader = PdfReader(io.BytesIO(pdf_data))
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()

#     # Create documents from extracted text
#     documents = SimpleDirectoryReader.from_string(text).load_data()
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

# @app.post("/embeddings")
# async def generate_embeddings(request: EmbeddingRequest, pdf_file: UploadFile = File(...)):
#     print(f"Received PDF for user: {request.username}")  # Add logging to check username

#     if not pdf_file:
#         raise HTTPException(status_code=400, detail="PDF file is required")

#     try:
#         # Process the PDF file and store embeddings
#         process_and_store_pdf(request.username, pdf_file)
#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# @app.post("/getResponse")
# async def llm_answers(username: str, query: str):
#     try:
#         # Retrieve user embeddings
#         embeddings_data = retrieve_embeddings(username)

#         # Reconstruct the index from stored embeddings
#         index = VectorStoreIndex.from_index_struct(VectorStoreIndex.IndexStruct.from_dict(embeddings_data), embed_model=embed_model)

#         # Setup retriever and query engine
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever)

#         # Perform query
#         response = query_engine.query(query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         return {"response": response.response, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import VectorStoreIndex
# import os
# from io import BytesIO
# import tempfile
# import fitz  # PyMuPDF

# app = FastAPI()

# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# # Function to process PDF and store embeddings
# def process_and_store_pdf(username: str, pdf_file: UploadFile):
#     # Read the PDF file content into memory
#     pdf_bytes = pdf_file.file.read()

#     # Create a temporary file to store the PDF bytes
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
#         temp_pdf_file.write(pdf_bytes)
#         temp_pdf_path = temp_pdf_file.name

#     # Process the PDF with PyMuPDF (fitz) for extracting text
#     doc = fitz.open(temp_pdf_path)
#     documents = []
    
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         text = page.get_text("text")
#         documents.append({"text": text})
    
#     # Create the embeddings index
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

#     # Clean up the temporary PDF file
#     os.remove(temp_pdf_path)

# @app.post("/embeddings")
# async def generate_embeddings(username: str = Form(...), pdf_file: UploadFile = File(...)):
#     # Check if the PDF file is provided
#     if not pdf_file:
#         raise HTTPException(status_code=400, detail="PDF file is required")

#     try:
#         # Process the PDF file and store embeddings
#         process_and_store_pdf(username, pdf_file)
#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Function to retrieve embeddings from MongoDB
# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# @app.post("/getResponse")
# async def llm_answers(username: str, query: str):
#     try:
#         # Retrieve user embeddings
#         embeddings_data = retrieve_embeddings(username)

#         # Reconstruct the index from stored embeddings
#         index = VectorStoreIndex.from_index_struct(VectorStoreIndex.IndexStruct.from_dict(embeddings_data), embed_model=embed_model)

#         # Setup retriever and query engine
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

#         # Perform query
#         response = query_engine.query(query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         return {"response": response.response, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import VectorStoreIndex, Document
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from pydantic import BaseModel
# import os
# from io import BytesIO
# import tempfile
# import fitz  # PyMuPDF

# app = FastAPI()

# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# # Function to process PDF and store embeddings
# def process_and_store_pdf(username: str, pdf_file: UploadFile):
#     # Read the PDF file content into memory
#     pdf_bytes = pdf_file.file.read()

#     # Create a temporary file to store the PDF bytes
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
#         temp_pdf_file.write(pdf_bytes)
#         temp_pdf_path = temp_pdf_file.name

#     # Process the PDF with PyMuPDF (fitz) for extracting text
#     doc = fitz.open(temp_pdf_path)
#     documents = []
    
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         text = page.get_text("text")
#         # Wrap the text in a Document object
#         documents.append(Document(text=text, doc_id=f"page_{page_num+1}"))
    
#     # Create the embeddings index
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

#     # Clean up the temporary PDF file
#     os.remove(temp_pdf_path)

# @app.post("/embeddings")
# async def generate_embeddings(username: str = Form(...), pdf_file: UploadFile = File(...)):
#     # Check if the PDF file is provided
#     if not pdf_file:
#         raise HTTPException(status_code=400, detail="PDF file is required")

#     try:
#         # Process the PDF file and store embeddings
#         process_and_store_pdf(username, pdf_file)
#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Function to retrieve embeddings from MongoDB
# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# # @app.post("/getResponse")
# # async def llm_answers(username: str, query: str):
# #     try:
# #         # Retrieve user embeddings
# #         embeddings_data = retrieve_embeddings(username)

# #         # Reconstruct the index from stored embeddings
# #         index = VectorStoreIndex.from_index_struct(VectorStoreIndex.IndexStruct.from_dict(embeddings_data), embed_model=embed_model)

# #         # Setup retriever and query engine
# #         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
# #         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

# #         # Perform query
# #         response = query_engine.query(query)
# #         context = "\n".join([node.text for node in response.source_nodes])

# #         return {"response": response.response, "context": context}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
# class QueryRequest(BaseModel):
#     username: str
#     query: str

# @app.post("/getResponse")
# # async def llm_answers(request: QueryRequest):
# #     try:
# #         # Retrieve user embeddings
# #         embeddings_data = retrieve_embeddings(request.username)

# #         # Reconstruct the index from stored embeddings
# #         index = VectorStoreIndex.from_index_struct(VectorStoreIndex.IndexStruct.from_dict(embeddings_data), embed_model=embed_model)

# #         # Setup retriever and query engine
# #         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
# #         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

# #         # Perform query
# #         response = query_engine.query(request.query)
# #         context = "\n".join([node.text for node in response.source_nodes])

# #         return {"response": response.response, "context": context}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
# async def llm_answers(request: QueryRequest):
#     try:
#         # Retrieve user embeddings
#         embeddings_data = retrieve_embeddings(request.username)

#         # Extract documents from the embeddings data (you may need to store the document texts separately)
#         documents = []
#         for doc_data in embeddings_data.get("documents", []):
#             documents.append(Document(text=doc_data["text"], doc_id=doc_data["doc_id"]))

#         # Rebuild the index from the documents
#         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#         # Setup retriever and query engine
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

#         # Perform query
#         response = query_engine.query(request.query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         return {"response": response.response, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings, VectorStoreIndex, Document
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from pydantic import BaseModel
# import os
# import tempfile
# import fitz  # PyMuPDF
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import login

# app = FastAPI()
# Settings.llm = None

# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# # Load base model from Hugging Face
# model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# base_model = AutoModelForCausalLM.from_pretrained(model_name,
#                                                  device_map="auto",
#                                                  trust_remote_code=False)

# # Apply PEFT fine-tuning
# config = PeftConfig.from_pretrained("Surya1502/suryagpt-ft")
# model = PeftModel.from_pretrained(base_model, "Surya1502/suryagpt-ft")

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# # Function to process PDF and store embeddings
# def process_and_store_pdf(username: str, pdf_file: UploadFile):
#     # Read the PDF file content into memory
#     pdf_bytes = pdf_file.file.read()

#     # Create a temporary file to store the PDF bytes
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
#         temp_pdf_file.write(pdf_bytes)
#         temp_pdf_path = temp_pdf_file.name

#     # Process the PDF with PyMuPDF (fitz) for extracting text
#     doc = fitz.open(temp_pdf_path)
#     documents = []
    
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         text = page.get_text("text")
#         # Wrap the text in a Document object
#         documents.append(Document(text=text, doc_id=f"page_{page_num+1}"))
    
#     # Create the embeddings index
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

#     # Clean up the temporary PDF file
#     os.remove(temp_pdf_path)

# @app.post("/embeddings")
# async def generate_embeddings(username: str = Form(...), pdf_file: UploadFile = File(...)):
#     # Check if the PDF file is provided
#     if not pdf_file:
#         raise HTTPException(status_code=400, detail="PDF file is required")

#     try:
#         # Process the PDF file and store embeddings
#         process_and_store_pdf(username, pdf_file)
#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Function to retrieve embeddings from MongoDB
# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# class QueryRequest(BaseModel):
#     username: str
#     query: str

# @app.post("/getResponse")
# async def llm_answers(request: QueryRequest):
#     try:
#         # Retrieve user embeddings
#         embeddings_data = retrieve_embeddings(request.username)

#         # Extract documents from the embeddings data
#         documents = []
#         for doc_data in embeddings_data.get("documents", []):
#             documents.append(Document(text=doc_data["text"], doc_id=doc_data["doc_id"]))

#         # Rebuild the index from the documents
#         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#         # Setup retriever and query engine
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

#         # Perform query
#         response = query_engine.query(request.query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         # Perform inference using the fine-tuned model
#         input_ids = tokenizer(request.query, return_tensors="pt").input_ids
#         outputs = model.generate(input_ids)
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return {"response": response_text, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings, VectorStoreIndex, Document
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from pydantic import BaseModel
# import os
# import tempfile
# import fitz  # PyMuPDF
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import login

# app = FastAPI()
# Settings.llm = None

# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# # Load a smaller, CPU-friendly model (distilgpt2)
# model_name = "distilgpt2"  # Replacing Mistral-7B-Instruct with distilgpt2
# base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=False)

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# # Function to process PDF and store embeddings
# def process_and_store_pdf(username: str, pdf_file: UploadFile):
#     # Read the PDF file content into memory
#     pdf_bytes = pdf_file.file.read()

#     # Create a temporary file to store the PDF bytes
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
#         temp_pdf_file.write(pdf_bytes)
#         temp_pdf_path = temp_pdf_file.name

#     # Process the PDF with PyMuPDF (fitz) for extracting text
#     doc = fitz.open(temp_pdf_path)
#     documents = []
    
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         text = page.get_text("text")
#         # Wrap the text in a Document object
#         documents.append(Document(text=text, doc_id=f"page_{page_num+1}"))
    
#     # Create the embeddings index
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

#     # Clean up the temporary PDF file
#     os.remove(temp_pdf_path)

# @app.post("/embeddings")
# async def generate_embeddings(username: str = Form(...), pdf_file: UploadFile = File(...)):
#     # Check if the PDF file is provided
#     if not pdf_file:
#         raise HTTPException(status_code=400, detail="PDF file is required")

#     try:
#         # Process the PDF file and store embeddings
#         process_and_store_pdf(username, pdf_file)
#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Function to retrieve embeddings from MongoDB
# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# class QueryRequest(BaseModel):
#     username: str
#     query: str

# @app.post("/getResponse")
# async def llm_answers(request: QueryRequest):
#     try:
#         # Retrieve user embeddings
#         embeddings_data = retrieve_embeddings(request.username)

#         # Extract documents from the embeddings data
#         documents = []
#         for doc_data in embeddings_data.get("documents", []):
#             documents.append(Document(text=doc_data["text"], doc_id=doc_data["doc_id"]))

#         # Rebuild the index from the documents
#         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#         # Setup retriever and query engine
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

#         # Perform query
#         response = query_engine.query(request.query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         # Perform inference using the fine-tuned model
#         input_ids = tokenizer(request.query, return_tensors="pt").input_ids
#         outputs = base_model.generate(input_ids)
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return {"response": response_text, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings, VectorStoreIndex, Document
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from pydantic import BaseModel
# import os
# import tempfile
# import fitz  # PyMuPDF
# from transformers import AutoModelForCausalLM, AutoTokenizer

# app = FastAPI()
# Settings.llm = None

# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# # Load a smaller model for local execution
# model_name = "distilgpt2"  # Using a smaller, more resource-efficient model
# base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", trust_remote_code=False)

# # Load tokenizer for the smaller model
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# # Function to process PDF and store embeddings
# def process_and_store_pdf(username: str, pdf_file: UploadFile):
#     # Read the PDF file content into memory
#     pdf_bytes = pdf_file.file.read()

#     # Create a temporary file to store the PDF bytes
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
#         temp_pdf_file.write(pdf_bytes)
#         temp_pdf_path = temp_pdf_file.name

#     # Process the PDF with PyMuPDF (fitz) for extracting text
#     doc = fitz.open(temp_pdf_path)
#     documents = []
    
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         text = page.get_text("text")
#         # Wrap the text in a Document object
#         documents.append(Document(text=text, doc_id=f"page_{page_num+1}"))
    
#     # Create the embeddings index
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

#     # Clean up the temporary PDF file
#     os.remove(temp_pdf_path)

# @app.post("/embeddings")
# async def generate_embeddings(username: str = Form(...), pdf_file: UploadFile = File(...)):
#     # Check if the PDF file is provided
#     if not pdf_file:
#         raise HTTPException(status_code=400, detail="PDF file is required")

#     try:
#         # Process the PDF file and store embeddings
#         process_and_store_pdf(username, pdf_file)
#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Function to retrieve embeddings from MongoDB
# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# class QueryRequest(BaseModel):
#     username: str
#     query: str

# # @app.post("/getResponse")
# # async def llm_answers(request: QueryRequest):
# #     try:
# #         # Retrieve user embeddings
# #         embeddings_data = retrieve_embeddings(request.username)

# #         # Extract documents from the embeddings data
# #         documents = []
# #         for doc_data in embeddings_data.get("documents", []):
# #             documents.append(Document(text=doc_data["text"], doc_id=doc_data["doc_id"]))

# #         # Rebuild the index from the documents
# #         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# #         # Setup retriever and query engine
# #         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
# #         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

# #         # Perform query
# #         response = query_engine.query(request.query)
# #         context = "\n".join([node.text for node in response.source_nodes])

# #         # Perform inference using the smaller model
# #         input_ids = tokenizer(request.query, return_tensors="pt").input_ids
# #         outputs = base_model.generate(input_ids)
# #         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# #         return {"response": response_text, "context": context}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
# @app.post("/getResponse")
# async def llm_answers(request: QueryRequest):
#     try:
#         # Retrieve user embeddings
#         embeddings_data = retrieve_embeddings(request.username)

#         # Extract documents from the embeddings data
#         documents = []
#         for doc_data in embeddings_data.get("documents", []):
#             documents.append(Document(text=doc_data["text"], doc_id=doc_data["doc_id"]))

#         # Rebuild the index from the documents
#         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#         # Setup retriever and query engine
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

#         # Perform query to retrieve relevant documents
#         response = query_engine.query(request.query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         # Set pad_token to eos_token if it's not already set
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token  # or you can set it to a custom pad token

#         # Prepare the input for the LLM
#         input_data = tokenizer(request.query, return_tensors="pt", padding=True, truncation=True)

#         # Generate attention mask if not provided
#         input_ids = input_data['input_ids']
#         attention_mask = input_data.get('attention_mask', None)

#         # If the attention mask is not provided, create it manually
#         if attention_mask is None:
#             attention_mask = (input_ids != tokenizer.pad_token_id).long()

#         # Perform inference using the model
#         outputs = base_model.generate(input_ids, attention_mask=attention_mask)
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return {"response": response_text, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings, VectorStoreIndex, Document
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from pydantic import BaseModel
# import os
# import tempfile
# import fitz  # PyMuPDF
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import login

# app = FastAPI()
# Settings.llm = None

# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# # Use GPT-2 model instead of Mistral
# model_name = "gpt2"  # GPT-2 model
# base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=False)

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# tokenizer.pad_token = tokenizer.eos_token
# # Function to process PDF and store embeddings
# def process_and_store_pdf(username: str, pdf_file: UploadFile):
#     # Read the PDF file content into memory
#     pdf_bytes = pdf_file.file.read()

#     # Create a temporary file to store the PDF bytes
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
#         temp_pdf_file.write(pdf_bytes)
#         temp_pdf_path = temp_pdf_file.name

#     # Process the PDF with PyMuPDF (fitz) for extracting text
#     doc = fitz.open(temp_pdf_path)
#     documents = []
    
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         text = page.get_text("text")
#         # Wrap the text in a Document object
#         documents.append(Document(text=text, doc_id=f"page_{page_num+1}"))
    
#     # Create the embeddings index
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

#     # Clean up the temporary PDF file
#     os.remove(temp_pdf_path)

# @app.post("/embeddings")
# async def generate_embeddings(username: str = Form(...), pdf_file: UploadFile = File(...)):
#     # Check if the PDF file is provided
#     if not pdf_file:
#         raise HTTPException(status_code=400, detail="PDF file is required")

#     try:
#         # Process the PDF file and store embeddings
#         process_and_store_pdf(username, pdf_file)
#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Function to retrieve embeddings from MongoDB
# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# class QueryRequest(BaseModel):
#     username: str
#     query: str

# @app.post("/getResponse")
# async def llm_answers(request: QueryRequest):
#     try:
#         # Retrieve user embeddings
#         embeddings_data = retrieve_embeddings(request.username)

#         # Extract documents from the embeddings data
#         documents = []
#         for doc_data in embeddings_data.get("documents", []):
#             documents.append(Document(text=doc_data["text"], doc_id=doc_data["doc_id"]))

#         # Rebuild the index from the documents
#         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#         # Setup retriever and query engine
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

#         # Perform query
#         response = query_engine.query(request.query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         # Prepare the input for GPT-2 model
#         input_data = tokenizer(request.query, return_tensors="pt", padding=True, truncation=True)
#         input_ids = input_data['input_ids']

#         # Generate attention mask if not provided
#         attention_mask = input_data.get('attention_mask', None)
#         if attention_mask is None:
#             attention_mask = (input_ids != tokenizer.pad_token_id).long()

#         # Perform inference using the GPT-2 model
#         outputs = base_model.generate(input_ids, attention_mask=attention_mask)
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return {"response": response_text, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from pymongo import MongoClient
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings, VectorStoreIndex, Document
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from pydantic import BaseModel
# import os
# import tempfile
# import fitz  # PyMuPDF
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from fastapi.middleware.cors import CORSMiddleware

# os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"

# app = FastAPI()
# Settings.llm = None
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Allow the frontend
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )
# # MongoDB setup
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "embeddings_db"
# COLLECTION_NAME = "user_embeddings"
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# # Embedding model setup
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# # Check for MPS device
# device = torch.device("cpu")

# # Load GPT-2 model from Hugging Face
# model_name = "distilgpt2"  # Replacing Mistral with GPT-2
# base_model = AutoModelForCausalLM.from_pretrained(model_name)
# base_model.to(device)

# # Apply PEFT fine-tuning (if applicable)
# # If you're using a fine-tuned model, load it accordingly
# # Example: 
# # config = PeftConfig.from_pretrained("YourFineTunedModel")
# # model = PeftModel.from_pretrained(base_model, "YourFineTunedModel")

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

# # Function to process PDF and store embeddings
# def process_and_store_pdf(username: str, pdf_file: UploadFile):
#     # Read the PDF file content into memory
#     pdf_bytes = pdf_file.file.read()

#     # Create a temporary file to store the PDF bytes
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
#         temp_pdf_file.write(pdf_bytes)
#         temp_pdf_path = temp_pdf_file.name

#     # Process the PDF with PyMuPDF (fitz) for extracting text
#     doc = fitz.open(temp_pdf_path)
#     documents = []
    
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)
#         text = page.get_text("text")
#         # Wrap the text in a Document object
#         documents.append(Document(text=text, doc_id=f"page_{page_num+1}"))
    
#     # Create the embeddings index
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#     # Store embeddings in MongoDB
#     embedding_data = {
#         "username": username,
#         "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
#     }
#     collection.replace_one({"username": username}, embedding_data, upsert=True)

#     # Clean up the temporary PDF file
#     os.remove(temp_pdf_path)

# @app.post("/embeddings")
# async def generate_embeddings(username: str = Form(...), pdf_file: UploadFile = File(...)):
#     # Check if the PDF file is provided
#     if not pdf_file:
#         raise HTTPException(status_code=400, detail="PDF file is required")

#     try:
#         # Process the PDF file and store embeddings
#         process_and_store_pdf(username, pdf_file)
#         return {"message": "Embeddings generated and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Function to retrieve embeddings from MongoDB
# def retrieve_embeddings(username: str):
#     record = collection.find_one({"username": username})
#     if not record:
#         raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
#     return record["embeddings"]

# class QueryRequest(BaseModel):
#     username: str
#     query: str

# # @app.post("/getResponse")
# # async def llm_answers(request: QueryRequest):
# #     try:
# #         # Retrieve user embeddings
# #         embeddings_data = retrieve_embeddings(request.username)

# #         # Extract documents from the embeddings data
# #         documents = []
# #         for doc_data in embeddings_data.get("documents", []):
# #             documents.append(Document(text=doc_data["text"], doc_id=doc_data["doc_id"]))

# #         # Rebuild the index from the documents
# #         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# #         # Setup retriever and query engine
# #         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
# #         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

# #         # Perform query
# #         response = query_engine.query(request.query)
# #         context = "\n".join([node.text for node in response.source_nodes])

# #         # Prepare the input for GPT-2 model
# #         input_data = tokenizer(request.query, return_tensors="pt", padding=True, truncation=True)
# #         input_ids = input_data['input_ids'].to(device)  # Move to MPS device

# #         # Generate attention mask if not provided
# #         attention_mask = input_data.get('attention_mask', None)
# #         if attention_mask is None:
# #             attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)  # Move to MPS device

# #         # Perform inference using the model
# #         outputs = base_model.generate(input_ids, attention_mask=attention_mask)
# #         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# #         return {"response": response_text, "context": context}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
# @app.post("/getResponse")
# async def llm_answers(request: QueryRequest):
#     try:
#         embeddings_data = retrieve_embeddings(request.username)
#         documents = [Document(text=doc_data["text"], doc_id=doc_data["doc_id"]) for doc_data in embeddings_data.get("documents", [])]
#         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])
#         response = query_engine.query(request.query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         # Combine context with the query
#         combined_input = f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:"
#         input_data = tokenizer(combined_input, return_tensors="pt", padding=True, truncation=True)
#         input_ids = input_data['input_ids'].to(device)
#         outputs = base_model.generate(input_ids, max_length=200)
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return {"response": response_text, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5100)


from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pymongo import MongoClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from pydantic import BaseModel
import os
import tempfile
import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware

os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"

app = FastAPI()
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow the frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "embeddings_db"
COLLECTION_NAME = "user_embeddings"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Embedding model setup
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Check for MPS device
device = torch.device("cpu")

# Load GPT-2 model from Hugging Face
model_name = "EleutherAI/pythia-2.8b"  # Replacing Mistral with GPT-2
#model_name = "gpt2"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.to(device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

# Function to process PDF and store embeddings
def process_and_store_pdf(username: str, pdf_file: UploadFile):
    # Read the PDF file content into memory
    pdf_bytes = pdf_file.file.read()

    # Create a temporary file to store the PDF bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
        temp_pdf_file.write(pdf_bytes)
        temp_pdf_path = temp_pdf_file.name

    # Process the PDF with PyMuPDF (fitz) for extracting text
    doc = fitz.open(temp_pdf_path)
    documents = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        # Wrap the text in a Document object
        documents.append(Document(text=text, doc_id=f"page_{page_num+1}"))

    # Create the embeddings index
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # Store embeddings in MongoDB
    embedding_data = {
        "username": username,
        "embeddings": index.index_struct.to_dict()  # Convert embeddings to serializable format
    }
    collection.replace_one({"username": username}, embedding_data, upsert=True)

    # Clean up the temporary PDF file
    os.remove(temp_pdf_path)

@app.post("/embeddings")
async def generate_embeddings(username: str = Form(...), pdf_file: UploadFile = File(...)):
    # Check if the PDF file is provided
    if not pdf_file:
        raise HTTPException(status_code=400, detail="PDF file is required")

    try:
        # Process the PDF file and store embeddings
        process_and_store_pdf(username, pdf_file)
        return {"message": "Embeddings generated and stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to retrieve embeddings from MongoDB
def retrieve_embeddings(username: str):
    record = collection.find_one({"username": username})
    if not record:
        raise HTTPException(status_code=404, detail="Embeddings not found for the specified user")
    return record["embeddings"]

class QueryRequest(BaseModel):
    username: str
    query: str

# @app.post("/getResponse")
# async def llm_answers(request: QueryRequest):
#     try:
#         embeddings_data = retrieve_embeddings(request.username)
#         documents = [Document(text=doc_data["text"], doc_id=doc_data["doc_id"]) for doc_data in embeddings_data.get("documents", [])]
#         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
#         query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])
#         response = query_engine.query(request.query)
#         context = "\n".join([node.text for node in response.source_nodes])

#         # Combine context with the query
#         combined_input = f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:"
#         input_data = tokenizer(combined_input, return_tensors="pt", padding=True, truncation=True)
#         input_ids = input_data['input_ids'].to(device)
#         attention_mask = input_data['attention_mask'].to(device)  # Explicitly pass attention mask

#         outputs = base_model.generate(input_ids, attention_mask=attention_mask, max_length=200)
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return {"response": response_text, "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/getResponse")
# async def llm_answers(request: QueryRequest):
#     try:
#         # Retrieve embeddings for the user
#         embeddings_data = retrieve_embeddings(request.username)
#         documents = [
#             Document(text=doc_data["text"], doc_id=doc_data["doc_id"])
#             for doc_data in embeddings_data.get("documents", [])
#         ]
        
#         # Create the index and retriever
#         index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
#         retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        
#         # Query engine with context retrieval
#         query_engine = RetrieverQueryEngine(
#             retriever=retriever,
#             node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.2)]
#         )
#         response = query_engine.query(request.query)

#         # Build context from source nodes
#         context = "\n".join([node.text for node in response.source_nodes])
#         print(f"Retrieved Context: {context}")
#         if not context.strip():
#             raise HTTPException(status_code=404, detail="No relevant context found")

#         # Combine context with query for LLM input
#         combined_input = f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:"
#         input_data = tokenizer(
#             combined_input,
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         )
#         input_ids = input_data["input_ids"].to(device)
#         attention_mask = input_data["attention_mask"].to(device)

#         # Generate answer with the language model
#         outputs = base_model.generate(
#             input_ids,
#             attention_mask=attention_mask,
#             max_length=200,
#             repetition_penalty=2.0
#         )
#         response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return {"response": response_text.strip(), "context": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/getResponse")
async def llm_answers(request: QueryRequest):
    try:
        # Retrieve embeddings for the user
        embeddings_data = retrieve_embeddings(request.username)
        documents = [
            Document(text=doc_data["text"], doc_id=doc_data["doc_id"])
            for doc_data in embeddings_data.get("documents", [])
        ]
        
        # Create the index and retriever
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        
        # Query engine with context retrieval
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.2)]
        )
        response = query_engine.query(request.query)
        print(f"Retrieved nodes: {response.source_nodes}")
        # Ensure source nodes are retrieved
        if not response.source_nodes:
            return {"response": "No relevant context found for the given query."}

        # Build context from source nodes
        top_k = min(3, len(response.source_nodes))  # Adjust '3' to your desired top_k value
        context_lines = ["Context:\n"]
        for i in range(top_k):
            context_lines.append(response.source_nodes[i].text)

        context = "\n\n".join(context_lines)
        print(f"Retrieved Context: {context}")

        # Combine context with query for LLM input
        combined_input = f"{context}\n\nQuestion: {request.query}\n\nAnswer:"
        input_data = tokenizer(
            combined_input,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = input_data["input_ids"].to(device)
        attention_mask = input_data["attention_mask"].to(device)

        # Generate answer with the language model
        outputs = base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=200,
            repetition_penalty=2.0
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"response": response_text.strip(), "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5100)
