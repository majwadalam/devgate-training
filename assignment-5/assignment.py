"""
Assignment 5: Building RAG with LangChain
Student Name: [Your Name Here]
Date: [Date]

Instructions:
1. Complete all the TODO sections below
2. Test your RAG system with the sample documents
3. Make sure to handle errors gracefully
4. Document your chunk size choices
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# TODO: Import necessary LangChain components
# Hint: You'll need document loaders, text splitters, embeddings, vector stores, and chains

# Load environment variables
load_dotenv()

class SimpleRAG:
    """A simple RAG (Retrieval-Augmented Generation) system using LangChain"""
    
    def __init__(self, documents_path: str = "documents"):
        """
        Initialize the RAG system
        
        Args:
            documents_path (str): Path to the directory containing documents
        """
        self.documents_path = documents_path
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        
        # TODO: Initialize your OpenAI API key
        # Hint: Use os.getenv() to get the API key from environment variables
        
        # TODO: Initialize the LLM (Language Model)
        # Hint: Use ChatOpenAI from langchain_openai
        
        # TODO: Initialize the embeddings
        # Hint: Use OpenAIEmbeddings from langchain_openai
        
        # Set up the RAG system
        self._setup_rag()
    
    def _load_documents(self) -> List[Any]:
        """
        Load documents from the documents directory
        
        Returns:
            List of loaded documents
        """
        documents = []
        
        # TODO: Implement document loading
        # Hint: Use TextLoader from langchain_community.document_loaders
        # Loop through all .txt files in the documents_path directory
        
        return documents
    
    def _split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of loaded documents
            
        Returns:
            List of document chunks
        """
        # TODO: Implement text splitting
        # Hint: Use RecursiveCharacterTextSplitter from langchain.text_splitter
        # Consider chunk_size=1000 and chunk_overlap=200 as starting points
        
        pass
    
    def _create_vectorstore(self, chunks: List[Any]):
        """
        Create a vector store from document chunks
        
        Args:
            chunks: List of document chunks
        """
        # TODO: Create and populate the vector store
        # Hint: Use Chroma from langchain_community.vectorstores
        # Use self.embeddings to create embeddings
        
        pass
    
    def _setup_retriever(self):
        """Set up the document retriever"""
        # TODO: Create a retriever from the vector store
        # Hint: Use vectorstore.as_retriever() with appropriate search parameters
        
        pass
    
    def _setup_chain(self):
        """Set up the RAG chain"""
        # TODO: Create a RAG chain
        # Hint: Use RetrievalQA from langchain.chains
        # Combine the retriever and LLM
        
        pass
    
    def _setup_rag(self):
        """Set up the complete RAG system"""
        try:
            print("Loading documents...")
            documents = self._load_documents()
            print(f"Loaded {len(documents)} documents")
            
            print("Splitting documents...")
            chunks = self._split_documents(documents)
            print(f"Created {len(chunks)} chunks")
            
            print("Creating vector store...")
            self._create_vectorstore(chunks)
            
            print("Setting up retriever...")
            self._setup_retriever()
            
            print("Setting up RAG chain...")
            self._setup_chain()
            
            print("RAG system ready!")
            
        except Exception as e:
            print(f"Error setting up RAG system: {e}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the RAG system
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict containing the answer and source information
        """
        if self.chain is None:
            return {"error": "RAG system not properly initialized"}
        
        try:
            # TODO: Use the chain to get an answer
            # Hint: Use self.chain.invoke() or similar method
            
            # TODO: Return the result in a structured format
            # Include both the answer and source documents if possible
            
            pass
            
        except Exception as e:
            return {"error": f"Error processing question: {e}"}

def main():
    """Main function to test the RAG system"""
    
    # Check if documents directory exists
    if not os.path.exists("documents"):
        print("Creating documents directory...")
        os.makedirs("documents")
        print("Please add some .txt files to the documents/ directory and run again.")
        return
    
    # Initialize RAG system
    try:
        rag = SimpleRAG()
        
        # Test questions
        test_questions = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms",
            "What are the applications of Python in AI?"
        ]
        
        print("\n" + "="*50)
        print("Testing RAG System")
        print("="*50)
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            print("-" * 40)
            
            result = rag.ask_question(question)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Answer: {result.get('answer', 'No answer provided')}")
                if 'sources' in result:
                    print(f"Sources: {result['sources']}")
        
        # Interactive mode
        print("\n" + "="*50)
        print("Interactive Mode (type 'quit' to exit)")
        print("="*50)
        
        while True:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            if not question:
                continue
                
            result = rag.ask_question(question)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nAnswer: {result.get('answer', 'No answer provided')}")
                if 'sources' in result:
                    print(f"Sources: {result['sources']}")
    
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        print("Make sure you have:")
        print("1. Set your OPENAI_API_KEY in the .env file")
        print("2. Installed all requirements: pip install -r requirements.txt")
        print("3. Added documents to the documents/ directory")

if __name__ == "__main__":
    main() 