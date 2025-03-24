# 🩺 Medical Chatbot with RAG Architecture

An AI-powered medical Q&A system using Llama 2, ChromaDB, and Sentence Transformers. Leverages Retrieval-Augmented Generation (RAG) to provide accurate responses from 16,000+ curated medical Q&A pairs.

## Features
- 🚀 Hybrid NLP pipeline with semantic search
- 💡 Local LLM inference with Llama 2-7B
- 📚 Context-aware responses using ChromaDB vector store
- 🐳 Dockerized services for scalability
- 🛠️ GPU-accelerated embeddings (CUDA support)

## Tech Stack
- **LLM**: Llama 2-7B (GGUF quantized)
- **Vector DB**: ChromaDB
- **Embeddings**: all-MiniLM-L6-v2
- **NLP**: Sentence Transformers, LangChain
- **API**: Ollama, Flask
- **UI**: Streamlit

## Prerequisites
- NVIDIA GPU (CUDA 11.8+ recommended)
- Docker & Docker Compose
- Python 3.10+
- Ollama installed locally

## 🛠️ Installation

1. Clone repository:
```bash
git clone https://github.com/<your-username>/medical-chatbot.git
cd medical-chatbot
Install dependencies:

bash
Copy
pip install -r requirements.txt
Download Llama 2-7B GGUF model:

bash
Copy
ollama pull llama2:7b-chat-q4_0
🐳 Docker Setup
Start services (Ollama + ChromaDB):

bash
Copy
docker-compose up -d
Verify containers:

bash
Copy
docker ps -a
🗄️ ChromaDB Setup
Run the ChromaDB initialization (from model_train.ipynb):

python
Copy
# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("medical_chatbot")

# Add your documents (from MedQuAD dataset)
# ... (run the document ingestion chunk from model_train)
Verify collection:

python
Copy
print(client.list_collections())
🚀 Running the Application
Start Streamlit UI:

bash
Copy
streamlit run Run_model.ipynb
Access the chatbot at http://localhost:8501

Project Structure
Copy
├── chroma_db/            # ChromaDB vector store
├── Model/                # Llama 2 GGUF models
├── Run_model.ipynb       # Main pipeline notebook
├── run_model_support.py  # Core RAG functions
├── requirements.txt      # Python dependencies
├── prompt.txt            # Llama 2 system prompt template
└── docker-compose.yml    # Ollama + ChromaDB configuration
Hardware Requirements
Minimum: 16GB RAM + 8GB VRAM (NVIDIA GPU)

Recommended: 32GB RAM + 16GB VRAM (RTX 3090/A100)

📚 Dataset
Uses 16,000+ hand-curated medical Q&A pairs from MedQuAD:

Symptoms analysis

Treatment protocols

Disease prevention

Medication information

Troubleshooting
Common issues:

CUDA Out of Memory:

python
Copy
# Reduce batch size in Run_model.ipynb
gpu_layers = min(30, int(gpu_mem * 3))  # Adjust multiplier
ChromaDB Connection Errors:

bash
Copy
rm -rf chroma_db/ && python model_train.py  # Reinitialize DB
Ollama API Timeouts:

bash
Copy
docker restart ollama  # Restart Ollama service
License
MIT License - See LICENSE
