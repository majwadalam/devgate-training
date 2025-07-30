# Assignment 5: Building RAG with LangChain

## Overview
Create a RAG (Retrieval-Augmented Generation) system that can answer questions about a specific document set using LangChain. This assignment focuses on document loading, text splitting, vector storage, and implementing a basic RAG chain.

## Learning Objectives
- Understand RAG architecture and components
- Work with LangChain document loaders and text splitters
- Implement vector storage using ChromaDB
- Build a basic RAG chain for question answering
- Handle document retrieval and context injection

## Setup Instructions

### 1. Create Virtual Environment

#### On Windows:
```bash
# Navigate to assignment-5 directory
cd assignment-5

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
# Navigate to assignment-5 directory
cd assignment-5

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# For windows (Git Bash)
source venv/Scripts/activate
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file in the assignment-5 directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Assignment Requirements

### Task Description
Create a RAG system that:

1. **Loads documents** from a local directory
2. **Splits text** into manageable chunks
3. **Creates embeddings** and stores them in a vector database
4. **Retrieves relevant chunks** based on user queries
5. **Generates answers** using the retrieved context

### Expected Features

#### Core Features (Required):
- [ ] Load text documents using LangChain loaders
- [ ] Split documents into chunks with appropriate overlap
- [ ] Create and store embeddings in ChromaDB
- [ ] Implement similarity search for document retrieval
- [ ] Build a RAG chain that combines retrieval and generation

#### Bonus Features (Optional):
- [ ] Support multiple document formats (PDF, TXT, MD)
- [ ] Add metadata filtering capabilities
- [ ] Implement conversation memory
- [ ] Add source citation in responses
- [ ] Create a simple CLI interface

### Sample Usage

#### Input Documents:
Place your documents in the `documents/` folder:
```
documents/
├── ai_basics.txt
├── python_guide.txt
└── machine_learning.txt
```

#### Example Interaction:
```python
# Question: "What is machine learning?"
# Expected Response: 
# "Machine learning is a subset of artificial intelligence that enables 
# computers to learn and make decisions from data without being explicitly 
# programmed for every task..."
# 
# Sources: machine_learning.txt (chunk 1)
```

## Files Structure

```
assignment-5/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── example_rag.py        # Working example solution
├── assignment.py         # Starter template for students
├── .env.example          # Environment variables template
├── documents/            # Sample documents for RAG
│   ├── ai_basics.txt
│   ├── python_guide.txt
│   └── ml_concepts.txt
└── venv/                 # Virtual environment (created by you)
```

## Getting Started

1. **Activate your virtual environment** (see setup instructions above)
2. **Set up your OpenAI API key** in the `.env` file
3. **Run the example script** to see expected behavior:
   ```bash
   python example_rag.py
   ```
4. **Examine the sample documents** in the `documents/` folder
5. **Complete the assignment** by editing `assignment.py`
6. **Test your solution** with different questions

## Key Concepts Covered

### 1. Document Loading
- Using LangChain's TextLoader
- Handling different file formats
- Loading multiple documents

### 2. Text Splitting
- RecursiveCharacterTextSplitter
- Chunk size and overlap considerations
- Preserving document structure

### 3. Vector Storage
- Creating embeddings with OpenAI
- Storing vectors in ChromaDB
- Similarity search implementation

### 4. RAG Chain
- Combining retrieval and generation
- Context injection into prompts
- Response formatting

## Submission Guidelines

### What to Submit:
- [ ] Your completed `assignment.py` file
- [ ] Any additional documents you used for testing
- [ ] Screenshots of your RAG system working
- [ ] Brief explanation of your chunk size choices

### Testing Your Solution:
```bash
# Run your assignment script
python assignment.py

# Test with different questions
python assignment.py --question "What is Python?"
```

## Common Issues & Solutions

### API Key Issues:
- **Problem**: OpenAI API key not found
- **Solution**: Check your `.env` file is properly configured

### Memory Issues:
- **Problem**: Large documents cause memory errors
- **Solution**: Use smaller chunk sizes or process documents in batches

### Retrieval Issues:
- **Problem**: Irrelevant chunks being retrieved
- **Solution**: Adjust chunk size, overlap, or similarity threshold

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [RAG Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

## Need Help?

- Review the `example_rag.py` for reference implementation
- Check the sample documents format
- Ask questions during the wrap-up session
- Test with simple documents first

Good luck building your first RAG system! 🤖📚 