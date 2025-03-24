# 🏥 Medical Chatbot with RAG & GPU-Accelerated Ollama

[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker)](https://www.docker.com)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

An end-to-end medical Q&A system combining RAG architecture with GPU-accelerated Llama 2 inference via Dockerized Ollama.

## Features
- 🐳 Full Dockerization (Ollama + ChromaDB)
- 🚀 NVIDIA GPU acceleration for LLM inference
- 🔍 Semantic search with ChromaDB
- 💊 Trained on 16k medical Q&A pairs
- 📈 Dynamic GPU resource allocation

## Tech Stack
| Component          | Technology               |
|---------------------|--------------------------|
| LLM Runtime         | Ollama (GPU-accelerated) |
| Vector DB           | ChromaDB                 |
| Embeddings Model    | all-MiniLM-L6-v2         |
| LLM Model           | Llama 2-7B-Chat (GGUF)   |
| Containerization    | Docker + NVIDIA Toolkit  |
| API Layer           | REST + Streamlit         |

## 🛠️ Prerequisites
1. NVIDIA GPU with **Driver 525+**
2. Docker Engine 24.0+
3. NVIDIA Container Toolkit
4. CUDA 11.8+ toolkit

# 🐳 Docker Compose Setup
```
version: '3.8'

services:
  ollama:
    image: ollama/ollama:0.1.31
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    networks:
      - medical-net
    runtime: nvidia

  chromadb:
    image: chromadb/chroma:0.4.15
    ports:
      - "8000:8000"
    networks:
      - medical-net
    volumes:
      - ./chroma_db:/chroma/chroma_db

volumes:
  ollama_models:

networks:
  medical-net:
    external: true
```
🔧 Ollama GPU Configuration
Pull Llama 2 model with GPU layers:
```
docker exec ollama ollama pull llama2:7b-chat-q4_0
```
Verify GPU acceleration:
```
docker exec ollama ollama ps
# Should show GPU layers in use
```

🗄️ ChromaDB Initialization
```
import chromadb

# Connect to Dockerized ChromaDB
client = chromadb.HttpClient(host="chromadb", port=8000)
collection = client.create_collection("medical_chatbot")

# Run your document ingestion code here
#install the chromaDB by just running the below chunk
# 🚀🚀🚀(From the ChromaDB setup chunk in model_train.ipynb)🚀🚀🚀Important to run the model
```

# Project Structure
```

Copy
├── docker-compose.yml       # GPU-Ollama + ChromaDB
├── chroma_db/              # Persistent vector store
├── Model/                  # Local model cache
├── Run_model.ipynb         # Main pipeline
├── model_train.ipynb       # DB initialization
├── requirements.txt        # Python dependencies
└── prompt.txt              # Llama 2 template
```
To Run
In command Prompt
```
streamlit run streamlit.py
```
