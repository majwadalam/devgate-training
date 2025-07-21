#!/usr/bin/env python3
"""
Assignment 4: Simple RAG Agent - Student Template
Name: [Your Name Here]
Date: [Date]
Description: Complete this script to build a simple document Q&A system using RAG
"""

import os
import json
import math
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# TODO: Import required libraries
# import openai
# from dotenv import load_dotenv


class SimpleVectorStore:
    """A simple in-memory vector store for document embeddings."""
    
    def __init__(self):
        """
        Initialize the vector store.
        
        TODO: Implement initialization
        - Set up storage for documents and embeddings
        - Initialize document counter
        """
        # TODO: Add your initialization code here
        pass
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a document to the vector store.
        
        Args:
            text (str): Document text
            metadata (Dict): Optional metadata
            
        Returns:
            str: Document ID
            
        TODO: Implement document addition
        - Generate unique document ID
        - Store document text and metadata
        - Return the document ID
        """
        # TODO: Your code here
        pass
    
    def add_embedding(self, doc_id: str, embedding: List[float]) -> None:
        """
        Add an embedding for a document.
        
        Args:
            doc_id (str): Document ID
            embedding (List[float]): Document embedding vector
            
        TODO: Store the embedding for the document
        """
        # TODO: Your code here
        pass
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1, vec2: Vectors to compare
            
        Returns:
            float: Similarity score (0-1)
            
        TODO: Implement cosine similarity calculation
        - Calculate dot product
        - Calculate vector magnitudes
        - Return cosine similarity
        """
        # TODO: Your code here
        # Hint: cosine_similarity = dot_product / (magnitude1 * magnitude2)
        pass
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding (List[float]): Query vector
            top_k (int): Number of results to return
            
        Returns:
            List[Tuple]: List of (doc_id, similarity_score, text)
            
        TODO: Implement similarity search
        - Calculate similarity with all documents
        - Sort by similarity score
        - Return top_k results
        """
        # TODO: Your code here
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dict: Statistics about stored documents
            
        TODO: Return stats like document count, etc.
        """
        # TODO: Your code here
        pass


class SimpleRAGAgent:
    """A simple RAG agent that can answer questions about documents."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the RAG agent.
        
        Args:
            api_key (str): OpenAI API key
            
        TODO: Implement initialization
        - Load environment variables
        - Set up OpenAI client
        - Initialize vector store
        - Set up statistics tracking
        """
        # TODO: Add your initialization code here
        pass
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI API.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
            
        TODO: Implement embedding generation
        - Call OpenAI embeddings API
        - Handle errors gracefully
        - Return embedding vector
        """
        # TODO: Your code here
        # Hint: Use openai.embeddings.create with model "text-embedding-3-small"
        pass
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
            
        Returns:
            List[str]: List of text chunks
            
        TODO: Implement text chunking
        - Split text into overlapping chunks
        - Handle edge cases (text shorter than chunk_size)
        - Return list of chunks
        """
        # TODO: Your code here
        pass
    
    def add_document(self, text: str, title: str = None) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            text (str): Document text
            title (str): Document title
            
        Returns:
            str: Document ID
            
        TODO: Implement document addition
        - Chunk the document text
        - Generate embeddings for each chunk
        - Store in vector store
        - Return document ID
        """
        print(f"📄 Adding document: {title or 'Untitled'}")
        
        # TODO: Your code here
        # Steps:
        # 1. Chunk the text
        # 2. For each chunk:
        #    - Add to vector store
        #    - Generate embedding
        #    - Store embedding
        
        pass
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query the knowledge base and generate an answer.
        
        Args:
            question (str): User question
            top_k (int): Number of relevant chunks to retrieve
            
        Returns:
            Dict: Response with answer and sources
            
        TODO: Implement RAG query
        - Generate embedding for question
        - Search for relevant documents
        - Create context from retrieved chunks
        - Generate answer using OpenAI API
        - Return response with sources
        """
        print(f"🔍 Searching for: {question}")
        
        # TODO: Your code here
        # Steps:
        # 1. Get embedding for question
        # 2. Search vector store for similar chunks
        # 3. Create context from top results
        # 4. Generate answer using OpenAI chat API
        # 5. Return formatted response
        
        pass
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using OpenAI API with context.
        
        Args:
            question (str): User question
            context (str): Retrieved context
            
        Returns:
            str: Generated answer
            
        TODO: Implement answer generation
        - Create prompt with context and question
        - Call OpenAI chat API
        - Return generated answer
        """
        # TODO: Your code here
        # Create a prompt like:
        # "Based on the following context, answer the question.
        #  Context: {context}
        #  Question: {question}
        #  Answer:"
        pass
    
    def load_documents_from_file(self, file_path: str) -> None:
        """
        Load documents from a text file.
        
        Args:
            file_path (str): Path to text file
            
        TODO: Implement file loading
        - Read file content
        - Add as document to knowledge base
        """
        # TODO: Your code here
        pass
    
    def interactive_mode(self):
        """
        Run the RAG agent in interactive mode.
        
        TODO: Implement interactive interface
        - Allow users to add documents
        - Let users ask questions
        - Display answers with sources
        - Handle commands like /help, /stats, /quit
        """
        print("🤖 Simple RAG Agent")
        print("=" * 30)
        print("Commands: '/add <text>' to add document, '/load <file>' to load file")
        print("Type '/help' for more commands, '/quit' to exit")
        print()
        
        # TODO: Implement the interactive loop
        # Hint: Similar to the chatbot assignment but with RAG features
        
        pass


def load_sample_documents() -> List[Dict[str, str]]:
    """Load sample documents for testing."""
    return [
        {
            "title": "Python Basics",
            "content": "Python is a high-level programming language. It's known for its simple syntax and readability. Python is widely used in web development, data science, artificial intelligence, and automation."
        },
        {
            "title": "Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. Common types include supervised learning, unsupervised learning, and reinforcement learning."
        },
        {
            "title": "RAG Systems",
            "content": "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval. RAG systems first retrieve relevant information from a knowledge base, then use that information to generate more accurate and informed responses."
        }
    ]


def main():
    """Main function to test your RAG agent."""
    parser = argparse.ArgumentParser(description="Simple RAG Agent")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--load", "-l", type=str, help="Load documents from file")
    parser.add_argument("--query", "-q", type=str, help="Ask a question")
    args = parser.parse_args()
    
    print("🚀 Assignment 4: Simple RAG Agent")
    print("=" * 40)
    
    try:
        # TODO: Initialize RAG agent
        # agent = SimpleRAGAgent()
        
        # TODO: Load sample documents
        # sample_docs = load_sample_documents()
        # for doc in sample_docs:
        #     agent.add_document(doc["content"], doc["title"])
        
        if args.interactive:
            # TODO: Run interactive mode
            # agent.interactive_mode()
            pass
        elif args.query:
            # TODO: Answer single question
            # response = agent.query(args.query)
            # print(f"Answer: {response.get('answer', 'No answer found')}")
            pass
        else:
            print("💡 Complete the TODO sections to make the RAG agent work!")
            print("💡 Try running with --interactive or --query 'your question'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have set up your OpenAI API key in the .env file")


# Test your implementation
if __name__ == "__main__":
    main()


# BONUS CHALLENGES (Optional):
# 1. Add support for different file formats (PDF, DOCX)
# 2. Implement persistent storage (save/load vector store)
# 3. Add metadata filtering for search
# 4. Create a web interface using Streamlit
# 5. Implement different chunking strategies
# 6. Add document summarization features


# TESTING CHECKLIST:
# □ Script runs without errors
# □ Can add documents to knowledge base
# □ Can generate embeddings successfully
# □ Can search for similar documents
# □ Can generate answers with context
# □ Interactive mode works
# □ Error handling works
# □ Code is well-commented and readable


# SUBMISSION:
# 1. Complete all TODO sections
# 2. Test with sample documents
# 3. Try asking various questions
# 4. Add your name and date at the top
# 5. Submit assignment.py and any test documents
# 6. Include a brief reflection on how RAG works 