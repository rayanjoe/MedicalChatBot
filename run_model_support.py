from sentence_transformers import SentenceTransformer
import chromadb
import requests
from dotenv import load_dotenv
def configure():
    load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("medical_chatbot")
def retrieve_relevant_chunks(query, top_k=3,):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True, device="cuda").tolist()

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_texts = results["documents"][0]  # Top-k documents
    return retrieved_texts

# Example Query for retrieving relevant chunks
"""
query = "What are the symptoms of diabetes?"
retrieved_texts = retrieve_relevant_chunks(query)
print("Relevant Chunks:", retrieved_texts)



local_model_path = "E:/Project/LLM_HEALTH_LOCAL_DATA/Model/Model/"

from ctransformers import AutoModelForCausalLM


gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
gpu_layers = min(50, int(gpu_mem * 5))  # Adjust dynamically

# Load GGUF model
model = AutoModelForCausalLM.from_pretrained(
    "E:/Project/LLM_HEALTH_LOCAL_DATA/Model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    gpu_layers=gpu_layers  # Adjust based on GPU memory
)

# Generate text
response = model("What are the symptoms of diabetes?", max_new_tokens=200)
print(response)

"""
def load_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
def rag_pipeline(query):
    query_embedding = embedding_model.encode(query).tolist()
    url = "http://localhost:11434/api/generate"
    prompt_text = load_prompt("prompt.txt")
    # Step 2: Retrieve relevant documents from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2 
    )
    retrieved_docs = results["documents"][0] if results["documents"] else []
    
    # Step 3: Prepare context for LLAMA 2
    if retrieved_docs:
        context = "\n".join(retrieved_docs)
    else:
        context = "No relevant documents were found in the database."

    # Step 4: Generate response using LLAMA 2
    prompt = f"[INST] <<SYS>>\n{prompt_text}\n<</SYS>>\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
    
    data = {
        "model": "llama2",
        "prompt": prompt,
        "stream": False
    }
    
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url=url, json=data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx status codes
        
        value = response.json()
        return value.get("response", "I couldn't find a response.")
    
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the API: {str(e)}"
    except KeyError:
        return "Error parsing the API response"