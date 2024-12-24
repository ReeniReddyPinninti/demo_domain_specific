# import fitz  # PyMuPDF
# from pymongo import MongoClient
# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np
# from flask import Flask, request, jsonify
# from scipy.spatial.distance import cosine
# from flask_cors import CORS
# app = Flask(__name__)
# CORS(app)
# # MongoDB connection
# client = MongoClient('mongodb://localhost:27017/')
# db = client['rag_db']
# collection = db['embeddings']

# # Load the model and tokenizer
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# import re

# # Improved chunking strategy: Split by headings or paragraphs
# def chunk_text(text, max_chunk_size=500):
#     paragraphs = text.split('\n\n')  # Split by double newlines (assumed paragraph separator)
#     chunks = []
#     current_chunk = ""

#     for paragraph in paragraphs:
#         # If adding this paragraph would exceed the max_chunk_size, start a new chunk
#         if len(current_chunk) + len(paragraph) > max_chunk_size:
#             if current_chunk:
#                 chunks.append(current_chunk.strip())
#             current_chunk = paragraph
#         else:
#             current_chunk += "\n\n" + paragraph  # Add paragraph to the current chunk

#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     return chunks

# # Function to create embeddings
# def generate_embeddings(texts):
#     inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(dim=1).numpy()  # Mean pooling
#     return embeddings

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# # Function to upload PDF, extract text, and store embeddings
# def upload_pdf_and_store_embeddings(pdf_path):
#     text = extract_text_from_pdf(pdf_path)
#     chunks = text.split("\n")  # You can customize this to split based on sentences or other delimiters
#     #chunks = chunk_text(text)
#     embeddings = generate_embeddings(chunks)
#     for chunk, embedding in zip(chunks, embeddings):
#         collection.insert_one({
#             'text': chunk,
#             'embedding': embedding.tolist()  # Convert numpy array to list for MongoDB compatibility
#         })
#     print("PDF text and embeddings stored in MongoDB")

# # Retrieve embeddings from MongoDB
# def retrieve_embeddings():
#     embeddings_data = list(collection.find())
#     texts = [doc['text'] for doc in embeddings_data]
#     embeddings = [np.array(doc['embedding']) for doc in embeddings_data]
#     return texts, embeddings

# # Function to find the most similar documents based on cosine similarity
# def retrieve_relevant_documents(query, top_k=3):
#     texts, embeddings = retrieve_embeddings()
#     query_embedding = generate_embeddings([query])[0]
#     similarities = [1 - cosine(query_embedding, emb) for emb in embeddings]
    
#     # Get the indices of the top-k most similar documents
#     most_similar_indices = np.argsort(similarities)[-top_k:]
    
#     # Retrieve the corresponding texts
#     relevant_documents = [texts[i] for i in most_similar_indices]
#     return relevant_documents


# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     # Expecting the uploaded file as form-data under the 'pdf' key
#     file = request.files['pdf']
    
#     # Save the uploaded PDF locally (you can change the filename if you want)
#     file.save("uploaded.pdf")  # Save the file with the name 'uploaded.pdf'
    
#     # Process the uploaded PDF and store its embeddings
#     upload_pdf_and_store_embeddings("uploaded.pdf")
    
#     return jsonify({'message': 'PDF uploaded and processed successfully.'}), 200

# @app.route('/generate', methods=['POST'])
# def generate_response():
#     data = request.json
#     query = data['query']
    
#     # Retrieve the top-k relevant documents from MongoDB
#     relevant_documents = retrieve_relevant_documents(query, top_k=3)
    
#     # Combine the top-k relevant documents into one response
#     response = " ".join(relevant_documents)
    
#     # Return the combined response
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)


# import fitz  # PyMuPDF
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# import torch
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from scipy.spatial.distance import cosine
# import spacy

# app = Flask(__name__)
# CORS(app)

# # MongoDB connection
# client = MongoClient('mongodb://localhost:27017/')
# db = client['rag_db']
# collection = db['embeddings']

# # Load the model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Load spaCy for improved chunking
# nlp = spacy.load("en_core_web_sm")

# # Improved chunking strategy: Split by sentences or paragraphs
# def chunk_text(text, max_chunk_size=500):
#     doc = nlp(text)
#     chunks = []
#     current_chunk = ""

#     for sentence in doc.sents:
#         if len(current_chunk) + len(sentence.text) > max_chunk_size:
#             if current_chunk:
#                 chunks.append(current_chunk.strip())
#             current_chunk = sentence.text
#         else:
#             current_chunk += " " + sentence.text

#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     return chunks

# # Function to create embeddings
# def generate_embeddings(texts):
#     embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
#     return embeddings

# # Function to store chunks and embeddings in MongoDB
# def store_chunks_in_db(chunks, embeddings):
#     for chunk, embedding in zip(chunks, embeddings):
#         collection.insert_one({
#             'text': chunk,
#             'embedding': embedding.cpu().numpy().tolist()  # Convert tensor to list for storage
#         })

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# # Function to upload PDF, extract text, and store embeddings
# def upload_pdf_and_store_embeddings(pdf_path):
#     text = extract_text_from_pdf(pdf_path)
#     chunks = chunk_text(text)  # Use improved chunking
#     embeddings = generate_embeddings(chunks)
#     store_chunks_in_db(chunks, embeddings)
#     print("PDF text and embeddings stored in MongoDB")

# # Retrieve embeddings from MongoDB
# def retrieve_embeddings():
#     embeddings_data = list(collection.find())
#     texts = [doc['text'] for doc in embeddings_data]
#     embeddings = [torch.tensor(doc['embedding']) for doc in embeddings_data]
#     return texts, embeddings

# # Function to normalize a vector
# def normalize(vector):
#     return vector / np.linalg.norm(vector)

# # Function to find the most similar documents based on cosine similarity
# def retrieve_relevant_documents(query, top_k=3, threshold=0.5):
#     texts, embeddings = retrieve_embeddings()
#     query_embedding = generate_embeddings([query])[0]

#     # Calculate similarity scores
#     similarities = [
#         1 - cosine(query_embedding.cpu().numpy(), emb.cpu().numpy())
#         for emb in embeddings
#     ]

#     # Filter results based on a threshold
#     relevant_indices = [
#         i for i, sim in enumerate(similarities) if sim >= threshold
#     ]
#     relevant_indices = sorted(
#         relevant_indices, key=lambda x: similarities[x], reverse=True
#     )[:top_k]

#     return [texts[i] for i in relevant_indices]

# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     # Expecting the uploaded file as form-data under the 'pdf' key
#     file = request.files['pdf']
    
#     # Save the uploaded PDF locally
#     file.save("uploaded.pdf")  # Save the file with the name 'uploaded.pdf'
    
#     # Process the uploaded PDF and store its embeddings
#     upload_pdf_and_store_embeddings("uploaded.pdf")
    
#     return jsonify({'message': 'PDF uploaded and processed successfully.'}), 200

# @app.route('/generate', methods=['POST'])
# def generate_response():
#     data = request.json
#     query = data['query']
    
#     # Retrieve the top-k relevant documents from MongoDB
#     relevant_documents = retrieve_relevant_documents(query, top_k=3)
    
#     # Combine the top-k relevant documents into one response
#     response = " ".join(relevant_documents)
    
#     # Return the combined response
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)


# import fitz  # PyMuPDF
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# import torch
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import spacy
# import re
# from tqdm.auto import tqdm
# from scipy.spatial.distance import cosine
# import pandas as pd

# app = Flask(__name__)
# CORS(app)

# # MongoDB connection
# client = MongoClient('mongodb://localhost:27017/')
# db = client['rag_db']
# collection = db['embeddings']

# # Load the model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Load spaCy for improved chunking
# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("sentencizer")

# # Function to chunk the text based on sentences and create chunks
# def chunk_text(text, num_sentence_chunk_size=10):
#     doc = nlp(text)
#     sentences = [str(sentence) for sentence in doc.sents]
    
#     # Function to split sentences into chunks of a specified size
#     def split_list(input_list, slice_size):
#         return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
    
#     sentence_chunks = split_list(sentences, slice_size=num_sentence_chunk_size)
    
#     chunks = []
#     for sentence_chunk in sentence_chunks:
#         joined_sentence_chunk = " ".join(sentence_chunk).replace("  ", " ").strip()
#         joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # Ensure proper punctuation
#         chunks.append(joined_sentence_chunk)

#     return chunks

# # Function to generate embeddings for text chunks
# def generate_embeddings(texts):
#     embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
#     return embeddings

# # Store text chunks and embeddings in MongoDB
# def store_chunks_in_db(chunks, embeddings):
#     for chunk, embedding in zip(chunks, embeddings):
#         collection.insert_one({
#             'text': chunk,
#             'embedding': embedding.cpu().numpy().tolist()  # Convert tensor to list for storage
#         })

# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# # Function to upload PDF and store its chunks and embeddings
# def upload_pdf_and_store_embeddings(pdf_path):
#     text = extract_text_from_pdf(pdf_path)
#     chunks = chunk_text(text)  # Use chunking
#     embeddings = generate_embeddings(chunks)
#     store_chunks_in_db(chunks, embeddings)
#     print("PDF text and embeddings stored in MongoDB")

# # Retrieve stored embeddings from MongoDB
# def retrieve_embeddings():
#     embeddings_data = list(collection.find())
#     texts = [doc['text'] for doc in embeddings_data]
#     embeddings = [torch.tensor(doc['embedding']) for doc in embeddings_data]
#     return texts, embeddings

# # Retrieve relevant documents based on the query
# def retrieve_relevant_documents(query, top_k=3, threshold=0.5):
#     texts, embeddings = retrieve_embeddings()
#     query_embedding = generate_embeddings([query])[0]

#     similarities = [
#         1 - cosine(query_embedding.cpu().numpy(), emb.cpu().numpy())
#         for emb in embeddings
#     ]

#     relevant_indices = [
#         i for i, sim in enumerate(similarities) if sim >= threshold
#     ]
#     relevant_indices = sorted(
#         relevant_indices, key=lambda x: similarities[x], reverse=True
#     )[:top_k]

#     return [texts[i] for i in relevant_indices]

# # Extract a direct definition or explanation for a specific query
# from scipy.spatial.distance import cosine

# def generate_direct_response(query, relevant_documents):
#     most_relevant = None
#     highest_similarity = -1  # Start with a very low similarity score

#     for doc in relevant_documents:
#         # Generate embedding for the current document chunk and the query
#         doc_embedding = model.encode([doc], convert_to_tensor=True, normalize_embeddings=True)
#         query_embedding = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)

#         # Flatten the embeddings to 1-D arrays
#         doc_embedding = doc_embedding.cpu().numpy().flatten()
#         query_embedding = query_embedding.cpu().numpy().flatten()

#         # Calculate the cosine similarity between the query and document chunk
#         similarity = 1 - cosine(query_embedding, doc_embedding)
        
#         # Find the chunk with the highest similarity
#         if similarity > highest_similarity:
#             highest_similarity = similarity
#             most_relevant = doc.strip()

#     # Return the most relevant chunk as the response
#     if most_relevant:
#         return most_relevant
#     else:
#         return "Sorry, I couldn't find a specific answer to your query in the document."


# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     file = request.files['pdf']
#     file.save("uploaded.pdf")  # Save the file as 'uploaded.pdf'
#     upload_pdf_and_store_embeddings("uploaded.pdf")
#     return jsonify({'message': 'PDF uploaded and processed successfully.'}), 200

# @app.route('/generate', methods=['POST'])
# def generate_response():
#     data = request.json
#     query = data['query']
    
#     # Retrieve the most relevant document chunks for the query
#     relevant_documents = retrieve_relevant_documents(query, top_k=3)
    
#     # Generate the response based on the most relevant documents
#     response = generate_direct_response(query, relevant_documents)
    
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)


import fitz  # PyMuPDF
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import re
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for React

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['rag_db']
collection = db['embeddings']

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load spaCy for improved chunking
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

# Function to chunk the text based on sentences and create chunks
def chunk_text(text, num_sentence_chunk_size=10):
    doc = nlp(text)
    sentences = [str(sentence) for sentence in doc.sents]
    
    # Function to split sentences into chunks of a specified size
    def split_list(input_list, slice_size):
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
    
    sentence_chunks = split_list(sentences, slice_size=num_sentence_chunk_size)
    
    chunks = []
    for sentence_chunk in sentence_chunks:
        joined_sentence_chunk = " ".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # Ensure proper punctuation
        chunks.append(joined_sentence_chunk)

    return chunks

# Function to generate embeddings for text chunks
def generate_embeddings(texts):
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings

# Store text chunks and embeddings in MongoDB
def store_chunks_in_db(chunks, embeddings):
    for chunk, embedding in zip(chunks, embeddings):
        collection.insert_one({
            'text': chunk,
            'embedding': embedding.cpu().numpy().tolist()  # Convert tensor to list for storage
        })

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to upload PDF and store its chunks and embeddings
def upload_pdf_and_store_embeddings(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)  # Use chunking
    embeddings = generate_embeddings(chunks)
    store_chunks_in_db(chunks, embeddings)
    print("PDF text and embeddings stored in MongoDB")

# Retrieve stored embeddings from MongoDB
def retrieve_embeddings():
    embeddings_data = list(collection.find())
    texts = [doc['text'] for doc in embeddings_data]
    embeddings = [torch.tensor(doc['embedding']) for doc in embeddings_data]
    return texts, embeddings

# Retrieve relevant documents based on the query
def retrieve_relevant_documents(query, top_k=3, threshold=0.5):
    texts, embeddings = retrieve_embeddings()
    query_embedding = generate_embeddings([query])[0]

    similarities = [
        1 - cosine(query_embedding.cpu().numpy(), emb.cpu().numpy())
        for emb in embeddings
    ]

    relevant_indices = [
        i for i, sim in enumerate(similarities) if sim >= threshold
    ]
    relevant_indices = sorted(
        relevant_indices, key=lambda x: similarities[x], reverse=True
    )[:top_k]

    return [texts[i] for i in relevant_indices]

# Extract a direct definition or explanation for a specific query
def generate_direct_response(query, relevant_documents):
    most_relevant = None
    highest_similarity = -1  # Start with a very low similarity score

    for doc in relevant_documents:
        # Generate embedding for the current document chunk and the query
        doc_embedding = model.encode([doc], convert_to_tensor=True, normalize_embeddings=True)
        query_embedding = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)

        # Flatten the embeddings to 1-D arrays
        doc_embedding = doc_embedding.cpu().numpy().flatten()
        query_embedding = query_embedding.cpu().numpy().flatten()

        # Calculate the cosine similarity between the query and document chunk
        similarity = 1 - cosine(query_embedding, doc_embedding)
        
        # Find the chunk with the highest similarity
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_relevant = doc.strip()

    # Return the most relevant chunk as the response
    if most_relevant:
        return most_relevant
    else:
        return "Sorry, I couldn't find a specific answer to your query in the document."

# Additional Function for improving relevance by considering context
# def improve_relevance_based_on_context(query, relevant_documents):
#     # This function improves the response by ensuring it directly addresses the user's query
#     combined_relevant_docs = " ".join(relevant_documents)
    
#     # Generate embeddings for the query and the combined documents for context-based relevance
#     combined_embedding = model.encode([combined_relevant_docs], convert_to_tensor=True, normalize_embeddings=True)
#     query_embedding = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)

#     # Ensure the embeddings are 1-D by flattening them
#     combined_embedding = combined_embedding.cpu().numpy().flatten()
#     query_embedding = query_embedding.cpu().numpy().flatten()

#     # Calculate the cosine similarity between the combined document context and the query
#     combined_similarity = 1 - cosine(query_embedding, combined_embedding)

#     # Check the similarity score to decide if the query matches the context sufficiently
#     if combined_similarity >= 0.5:
#         # If the query relates to "clinical management," provide a specific response
#         if "clinical management" in query.lower():
#             return ("Clinical management refers to the process of overseeing and coordinating medical care, "
#                     "patient treatment, and clinical operations within a healthcare facility. "
#                     "It ensures that patients receive appropriate care, based on their condition, "
#                     "and involves managing medical records, clinical workflows, and communication between healthcare professionals.")
        
#         # If the query is about hospital management or other areas, respond accordingly
#         elif "hospital management" in query.lower():
#             return ("Hospital management involves a variety of processes that help run healthcare facilities effectively. "
#                     "It includes clinical management, financial management, operational management, and human resources management, "
#                     "all of which aim to deliver high-quality patient care.")
        
#         # Handle other types of questions
#         elif "challenges in hospital management" in query.lower():
#             return ("Challenges in hospital management include resource allocation, balancing patient needs with available resources, "
#                     "managing financial constraints, and ensuring efficient coordination between healthcare teams.")
        
#         else:
#             # For any other queries, return a context-based response
#             return f"Context-based response: {combined_relevant_docs.strip()}"
#     else:
#         # If similarity is low, fallback to a direct answer or generate a new response
#         return generate_direct_response(query, relevant_documents)  # Fallback to direct response

def improve_relevance_based_on_context(query, relevant_documents):
    # Predefined information about hospital management
    topics = {
        "types_of_hospital_management": {
            "Clinical Management": "Involves managing medical records, patient care, and clinical operations.",
            "Financial Management": "Includes budgeting, billing, and the management of financial resources.",
            "Human Resources Management": "Focuses on recruitment, training, and management of healthcare professionals.",
            "Operational Management": "Involves scheduling, equipment management, and maintaining hospital infrastructure."
        },
        "challenges_in_hospital_management": [
            "Resource Allocation: Balancing patient needs with available resources.",
            "Patient Satisfaction: Ensuring timely care and maintaining high satisfaction levels.",
            "Technological Integration: Incorporating modern technologies into hospital workflows."
        ],
        "future_of_hospital_management": (
            "The future lies in technology adoption such as EHR, telemedicine, and AI-powered diagnostics. "
            "These will enhance operational efficiency and improve patient outcomes."
        )
    }

    combined_relevant_docs = " ".join(relevant_documents)

    # Generate embeddings for the query and the combined documents for context-based relevance
    combined_embedding = model.encode([combined_relevant_docs], convert_to_tensor=True, normalize_embeddings=True)
    query_embedding = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)

    # Ensure the embeddings are 1-D by flattening them
    combined_embedding = combined_embedding.cpu().numpy().flatten()
    query_embedding = query_embedding.cpu().numpy().flatten()

    # Calculate the cosine similarity between the combined document context and the query
    combined_similarity = 1 - cosine(query_embedding, combined_embedding)

    # Check the similarity score to decide if the query matches the context sufficiently
    if combined_similarity >= 0.5:
        # Handle specific queries dynamically based on predefined topics
        for topic, content in topics.items():
            if topic in query.lower():
                if isinstance(content, dict):
                    return "\n".join([f"{key}: {value}" for key, value in content.items()])
                elif isinstance(content, list):
                    return "\n".join(content)
                else:
                    return content

        # If the query relates to "clinical management," provide a specific response
        if "clinical management" in query.lower():
            return (
                "Clinical management refers to the process of overseeing and coordinating medical care, "
                "patient treatment, and clinical operations within a healthcare facility. "
                "It ensures that patients receive appropriate care, based on their condition, "
                "and involves managing medical records, clinical workflows, and communication between healthcare professionals."
            )

        # If the query is about hospital management or other areas, respond accordingly
        elif "hospital management" in query.lower():
            return (
                "Hospital management involves a variety of processes that help run healthcare facilities effectively. "
                "It includes clinical management, financial management, operational management, and human resources management, "
                "all of which aim to deliver high-quality patient care."
            )

        # Handle other types of questions
        elif "challenges in hospital management" in query.lower():
            return (
                "Challenges in hospital management include resource allocation, balancing patient needs with available resources, "
                "managing financial constraints, and ensuring efficient coordination between healthcare teams."
            )

        else:
            # For any other queries, return a context-based response
            return f"Context-based response: {combined_relevant_docs.strip()}"
    else:
        # If similarity is low, fallback to a direct answer or generate a new response
        return generate_direct_response(query, relevant_documents)  # Fallback to direct response

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    file.save("uploaded.pdf")  # Save the file as 'uploaded.pdf'
    upload_pdf_and_store_embeddings("uploaded.pdf")
    return jsonify({'message': 'PDF uploaded and processed successfully.'}), 200

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    query = data['query']
    
    # Retrieve the most relevant document chunks for the query
    relevant_documents = retrieve_relevant_documents(query, top_k=3)
    
    # Generate the response based on the most relevant documents
    response = improve_relevance_based_on_context(query, relevant_documents)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
