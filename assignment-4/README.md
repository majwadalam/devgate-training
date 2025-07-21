# Assignment 4: Simple RAG Agent

## Overview
Build a simple Retrieval-Augmented Generation (RAG) system that can answer questions about documents using OpenAI embeddings and chat completion. This assignment introduces the core concepts of vector search, document chunking, and context-aware AI responses.

## Learning Objectives
- Understand RAG architecture and concepts
- Work with OpenAI embeddings API
- Implement basic vector similarity search
- Build document chunking and retrieval systems
- Create context-aware AI responses

## Setup Instructions

### 1. Create Virtual Environment

#### On Windows:
```bash
# Navigate to assignment-4 directory
cd assignment-4

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
# Navigate to assignment-4 directory
cd assignment-4

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# For windows (Git Bash)
source venv/Scripts/activate

# For windows (Powershell)
./venv/Scripts/activate
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Set Up API Access
```bash
# Create .env file for your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

**Important:** Replace `your_api_key_here` with your actual OpenAI API key.

### 4. Verify Installation
```bash
# Run example script to see RAG in action
python example_script.py
```

## Assignment Requirements

### Task Description
Create a RAG system that:

1. **Stores documents in a vector database** using embeddings
2. **Chunks long documents** into manageable pieces
3. **Searches for relevant content** using cosine similarity
4. **Generates informed answers** using retrieved context
5. **Provides an interactive interface** for document management and queries

### Expected Features

#### Core Features (Required):
- [ ] Document chunking with overlap
- [ ] OpenAI embeddings generation
- [ ] Vector similarity search (cosine similarity)
- [ ] Context-aware answer generation
- [ ] Interactive command-line interface
- [ ] Document loading from text files

#### Bonus Features (Optional):
- [ ] Support for different file formats (PDF, DOCX)
- [ ] Persistent vector store (save/load)
- [ ] Web interface using Streamlit
- [ ] Metadata filtering for searches
- [ ] Document summarization

### Sample Interaction

```
🤖 Simple RAG Agent
==================
Commands: '/add <text>' to add document, '/load <file>' to load file
Type '/help' for more commands, '/quit' to exit

You: /add Python is a programming language known for its simplicity and readability.

📄 Adding document: Untitled
   Split into 1 chunks
   ✅ Processed chunk 1/1
✅ Document added successfully (1 chunks)

You: What is Python?

🔍 Searching for: What is Python?
✅ Found 1 relevant sources
📊 Top similarity: 0.85

🤖 Answer: Python is a programming language known for its simplicity and readability.

📚 Sources:
  1. Untitled (similarity: 0.85)

You: /stats

📊 RAG Agent Statistics:
  Documents added: 1
  Total chunks: 1
  Queries processed: 1
  Embeddings generated: 2
  Total cost: $0.0003

You: /quit
👋 Goodbye!
```

## RAG System Architecture

### 1. Document Processing
```
Text Document → Chunking → Embeddings → Vector Store
```

### 2. Query Processing
```
User Question → Embedding → Similarity Search → Context Retrieval → AI Answer
```

### Key Components:

#### SimpleVectorStore
- Stores document chunks and their embeddings
- Implements cosine similarity search
- Manages document metadata

#### SimpleRAGAgent
- Orchestrates the RAG pipeline
- Handles OpenAI API interactions
- Manages user interface

## Files Structure

```
assignment-4/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── example_script.py      # Working example solution
├── assignment.py          # Starter template for students
├── sample_documents.txt   # Sample documents for testing
├── .env                   # API key (create this)
└── venv/                  # Virtual environment (created by you)
```

## Getting Started

1. **Activate your virtual environment** (see setup instructions above)
2. **Set up your API key** in the `.env` file
3. **Run the example script** to see expected behavior:
   ```bash
   python example_script.py
   ```
4. **Try the interactive mode**:
   ```bash
   python example_script.py --interactive
   ```
5. **Complete the assignment** by editing `assignment.py`

## How RAG Works

### Step 1: Document Ingestion
1. Split documents into chunks (with overlap)
2. Generate embeddings for each chunk
3. Store chunks and embeddings in vector database

### Step 2: Query Processing
1. Generate embedding for user question
2. Find most similar document chunks (cosine similarity)
3. Retrieve top-k relevant chunks as context

### Step 3: Answer Generation
1. Combine retrieved chunks into context
2. Create prompt with context + question
3. Generate answer using LLM
4. Return answer with source citations

## Key Concepts

### Embeddings
- Vector representations of text
- Similar texts have similar embeddings
- Enable semantic search beyond keyword matching

### Chunking
- Split long documents into smaller pieces
- Overlap ensures context preservation
- Balance between detail and relevance

### Vector Similarity
- Cosine similarity measures vector direction
- Range: -1 (opposite) to 1 (identical)
- Threshold for relevance filtering

## Submission Guidelines

### What to Submit:
- [ ] Your completed `assignment.py` file
- [ ] Sample documents you tested with
- [ ] Screenshots of your RAG system in action
- [ ] Brief reflection on how RAG works

### Testing Your Solution:
```bash
# Run your RAG agent
python assignment.py --interactive

# Test with custom documents
python assignment.py --load sample_documents.txt

# Ask specific questions
python assignment.py --query "What is machine learning?"
```

## Common Issues & Solutions

### High API Costs:
- **Problem**: Embedding generation can be expensive
- **Solution**: Use smaller chunks, cache embeddings

### Poor Retrieval:
- **Problem**: Irrelevant chunks retrieved
- **Solution**: Improve chunking strategy, adjust similarity threshold

### Slow Performance:
- **Problem**: Linear search through all embeddings
- **Solution**: Use approximate nearest neighbor (FAISS, ChromaDB)

## Additional Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/)
- [RAG System Design](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)
- [Chunking Strategies](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

## Need Help?

- Review the `example_script.py` for reference implementation
- Check the `sample_documents.txt` for example content
- Ask questions during the wrap-up session
- Collaborate with peers on Discord/Slack

Good luck building your first RAG system! 🚀 