"""
Assignment 5: Example RAG Implementation
This is a working example of a basic RAG system using LangChain.
Students can reference this while implementing their own solution.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables
load_dotenv()

class ExampleRAG:
    """Example RAG system implementation"""
    
    def __init__(self, documents_path: str = "documents"):
        """Initialize the RAG system"""
        self.documents_path = documents_path
        
        # Initialize OpenAI components
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize components
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        
        # Set up the RAG system
        self._setup_rag()
    
    def _load_documents(self) -> List[Document]:
        """Load documents from the documents directory"""
        documents = []
        
        if not os.path.exists(self.documents_path):
            print(f"Documents directory '{self.documents_path}' not found!")
            return documents
        
        # Load all .txt files
        for filename in os.listdir(self.documents_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.documents_path, filename)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def _create_vectorstore(self, chunks: List[Document]):
        """Create vector store from document chunks"""
        if not chunks:
            print("No chunks to create vector store!")
            return
        
        # Create Chroma vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Persist the database
        self.vectorstore.persist()
    
    def _setup_retriever(self):
        """Set up the document retriever"""
        if self.vectorstore is None:
            print("Vector store not initialized!")
            return
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    
    def _setup_chain(self):
        """Set up the RAG chain"""
        if self.retriever is None:
            print("Retriever not initialized!")
            return
        
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def _setup_rag(self):
        """Set up the complete RAG system"""
        try:
            print("Setting up RAG system...")
            
            # Load documents
            documents = self._load_documents()
            if not documents:
                print("No documents loaded. Please add .txt files to the documents/ directory.")
                return
            
            print(f"Loaded {len(documents)} documents")
            
            # Split documents
            chunks = self._split_documents(documents)
            print(f"Created {len(chunks)} chunks")
            
            # Create vector store
            self._create_vectorstore(chunks)
            print("Vector store created")
            
            # Set up retriever
            self._setup_retriever()
            print("Retriever configured")
            
            # Set up chain
            self._setup_chain()
            print("RAG chain ready!")
            
        except Exception as e:
            print(f"Error setting up RAG system: {e}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question to the RAG system"""
        if self.chain is None:
            return {
                "error": "RAG system not properly initialized. Please check your setup."
            }
        
        try:
            # Get response from RAG chain
            response = self.chain.invoke({"query": question})
            
            # Format sources
            sources = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
            
            return {
                "question": question,
                "answer": response.get("result", "No answer generated"),
                "sources": sources
            }
            
        except Exception as e:
            return {"error": f"Error processing question: {e}"}

def main():
    """Main function to demonstrate the RAG system"""
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return
    
    # Initialize RAG system
    try:
        rag = ExampleRAG()
        
        # Test questions
        test_questions = [
            "What is artificial intelligence?",
            "How is Python used in AI development?",
            "What are the types of machine learning?",
            "What are the challenges in AI development?"
        ]
        
        print("\n" + "="*60)
        print("🤖 RAG SYSTEM DEMONSTRATION")
        print("="*60)
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            print("-" * 50)
            
            result = rag.ask_question(question)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            print(f"✅ Answer: {result['answer']}")
            
            if result.get('sources'):
                print(f"\n📚 Sources ({len(result['sources'])} found):")
                for i, source in enumerate(result['sources'][:2], 1):
                    print(f"  {i}. {source['content']}")
            
            print()
        
        # Interactive mode
        print("\n" + "="*60)
        print("🔄 INTERACTIVE MODE")
        print("Type your questions below (or 'quit' to exit)")
        print("="*60)
        
        while True:
            question = input("\n💬 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            result = rag.ask_question(question)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"\n🎯 Answer: {result['answer']}")
                
                if result.get('sources'):
                    print(f"\n📖 Sources: {len(result['sources'])} documents found")
    
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have set OPENAI_API_KEY in your .env file")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Add .txt files to the documents/ directory")

if __name__ == "__main__":
    main() 