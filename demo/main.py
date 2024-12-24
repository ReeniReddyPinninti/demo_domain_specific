from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI()

class EmbeddingRequest(BaseModel):
    email: str
    data: str

class QueryRequest(BaseModel):
    query: str
    username: str



@app.post("/embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    email = request.email
    data = request.data

   
    return {"message": "Embeddings generation process completed"}

@app.post("/getResponse")
async def llm_answers(request: QueryRequest):
    query = request.query
    username = request.username

    

    
       
    return {"message": "hello"}

@app.post("/getResponseNeo")
async def Neollm_answers(request: QueryRequest):
    query = request.query
    username = request.username

    
    return {"message":  'hello'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5100)