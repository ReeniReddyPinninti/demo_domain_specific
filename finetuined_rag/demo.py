from flask import Flask, request, jsonify
import os
import tempfile
import pymongo
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"

app = Flask(__name__)

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "embeddings_db"
COLLECTION_NAME = "user_embeddings"
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Model and tokenizer setup
MODEL_NAME = "distilbert-base-uncased"  # Lightweight model for CPU
device = torch.device("cpu")

try:
    # Load lightweight model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    embedding_model = embedding_model.to(device)

    llm_model_name = "gpt2"  # Lightweight causal LM for CPU
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
    llm_model = llm_model.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load the models or tokenizers: {e}")


# Function to process PDF and store embeddings
def process_and_store_pdf(username, pdf_file):
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
        temp_pdf_file.write(pdf_file.read())
        temp_pdf_path = temp_pdf_file.name

    try:
        # Process the PDF to extract text
        doc = fitz.open(temp_pdf_path)
        documents = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            documents.append({"text": text, "doc_id": f"page_{page_num+1}"})

        # Placeholder for embedding creation
        embedding_data = {"username": username, "documents": documents}

        # Store embeddings in MongoDB
        collection.replace_one({"username": username}, embedding_data, upsert=True)
    finally:
        # Clean up the temporary PDF file
        os.remove(temp_pdf_path)


@app.route("/embeddings", methods=["POST"])
def generate_embeddings():
    username = request.form.get("username")
    pdf_file = request.files.get("pdf_file")

    if not username or not pdf_file:
        return jsonify({"error": "Username and PDF file are required"}), 400

    try:
        process_and_store_pdf(username, pdf_file)
        return jsonify({"message": "Embeddings generated and stored successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to retrieve embeddings from MongoDB
def retrieve_embeddings(username):
    record = collection.find_one({"username": username})
    if not record:
        return None
    return record["documents"]


@app.route("/getResponse", methods=["POST"])
def llm_answers():
    data = request.get_json()
    username = data.get("username")
    query = data.get("query")

    if not username or not query:
        return jsonify({"error": "Username and query are required"}), 400

    try:
        documents = retrieve_embeddings(username)
        if not documents:
            return jsonify({"error": "Embeddings not found for the specified user"}), 404

        # Combine documents into context
        context = "\n".join([doc["text"] for doc in documents])

        # Prepare input for the model
        combined_input = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        input_data = tokenizer(combined_input, return_tensors="pt", padding=True, truncation=True)

        # Generate response
        input_ids = input_data['input_ids'].to(device)
        outputs = llm_model.generate(
            input_ids=input_ids,
            max_new_tokens=200,  # Limit generated tokens
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": f"Error during LLM answer generation: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100)