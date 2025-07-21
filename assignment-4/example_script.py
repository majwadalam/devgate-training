#!/usr/bin/env python3
"""
Assignment 4: Simple RAG Agent Example Script
Author: GenAI Bootcamp
Description: Complete implementation of a simple document Q&A system using RAG
"""

import os
import json
import math
import argparse
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import openai
from dotenv import load_dotenv


class SimpleVectorStore:
    """A simple in-memory vector store for document embeddings."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.documents = {}  # doc_id -> {text, metadata, timestamp}
        self.embeddings = {}  # doc_id -> embedding vector
        self.doc_counter = 0
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the vector store."""
        doc_id = f"doc_{self.doc_counter}_{uuid.uuid4().hex[:8]}"
        self.doc_counter += 1
        
        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        return doc_id
    
    def add_embedding(self, doc_id: str, embedding: List[float]) -> None:
        """Add an embedding for a document."""
        if doc_id not in self.documents:
            raise ValueError(f"Document {doc_id} not found")
        self.embeddings[doc_id] = embedding
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Search for similar documents."""
        if not self.embeddings:
            return []
        
        similarities = []
        for doc_id, embedding in self.embeddings.items():
            similarity = self.cosine_similarity(query_embedding, embedding)
            text = self.documents[doc_id]["text"]
            similarities.append((doc_id, similarity, text))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_documents": len(self.documents),
            "documents_with_embeddings": len(self.embeddings),
            "embedding_dimension": len(next(iter(self.embeddings.values()))) if self.embeddings else 0
        }


class SimpleRAGAgent:
    """A simple RAG agent that can answer questions about documents."""
    
    def __init__(self, api_key: str = None):
        """Initialize the RAG agent."""
        # Load environment variables
        load_dotenv()
        
        # Set up OpenAI client
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize vector store
        self.vector_store = SimpleVectorStore()
        
        # Statistics tracking
        self.total_queries = 0
        self.total_documents_added = 0
        self.total_embeddings_generated = 0
        self.total_cost = 0.0
        
        # Embedding model and costs
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"
        self.embedding_cost_per_1k = 0.00002  # $0.00002 per 1K tokens
        self.chat_cost_per_1k_input = 0.0015
        self.chat_cost_per_1k_output = 0.006
        
        print(f"✅ RAG Agent initialized with embedding model: {self.embedding_model}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API."""
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            # Calculate cost (approximate token count)
            token_count = len(text.split()) * 1.3  # Rough approximation
            cost = (token_count / 1000) * self.embedding_cost_per_1k
            self.total_cost += cost
            self.total_embeddings_generated += 1
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    space = text.rfind(' ', start, end)
                    if space > start + chunk_size // 2:
                        end = space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_document(self, text: str, title: str = None) -> str:
        """Add a document to the knowledge base."""
        print(f"📄 Adding document: {title or 'Untitled'}")
        
        # Chunk the text
        chunks = self.chunk_text(text)
        print(f"   Split into {len(chunks)} chunks")
        
        document_ids = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Add chunk to vector store
            metadata = {
                "title": title or "Untitled",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": "user_input"
            }
            
            doc_id = self.vector_store.add_document(chunk, metadata)
            
            # Generate and store embedding
            try:
                embedding = self.get_embedding(chunk)
                self.vector_store.add_embedding(doc_id, embedding)
                document_ids.append(doc_id)
                print(f"   ✅ Processed chunk {i+1}/{len(chunks)}")
            except Exception as e:
                print(f"   ❌ Failed to process chunk {i+1}: {e}")
        
        self.total_documents_added += 1
        print(f"✅ Document added successfully ({len(document_ids)} chunks)")
        
        return document_ids[0] if document_ids else None
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the knowledge base and generate an answer."""
        print(f"🔍 Searching for: {question}")
        self.total_queries += 1
        
        try:
            # Get embedding for question
            question_embedding = self.get_embedding(question)
            
            # Search for relevant documents
            results = self.vector_store.search(question_embedding, top_k)
            
            if not results:
                return {
                    "answer": "I don't have any relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Create context from top results
            context_parts = []
            sources = []
            
            for doc_id, similarity, text in results:
                context_parts.append(f"[Source {len(sources)+1}]: {text}")
                metadata = self.vector_store.documents[doc_id]["metadata"]
                sources.append({
                    "doc_id": doc_id,
                    "similarity": similarity,
                    "title": metadata.get("title", "Unknown"),
                    "chunk_index": metadata.get("chunk_index", 0)
                })
            
            context = "\n\n".join(context_parts)
            
            # Generate answer
            answer = self.generate_answer(question, context)
            
            # Calculate confidence based on top similarity score
            confidence = results[0][1] if results else 0.0
            
            print(f"✅ Found {len(results)} relevant sources")
            print(f"📊 Top similarity: {confidence:.2f}")
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "context_used": context
            }
            
        except Exception as e:
            print(f"❌ Error during query: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using OpenAI API with context."""
        system_prompt = """You are a helpful assistant that answers questions based on provided context. 
Use only the information in the context to answer questions. If the context doesn't contain enough 
information to answer the question, say so clearly. Be concise but informative."""
        
        user_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = openai.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens / 1000 * self.chat_cost_per_1k_input + 
                   output_tokens / 1000 * self.chat_cost_per_1k_output)
            self.total_cost += cost
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def load_documents_from_file(self, file_path: str) -> None:
        """Load documents from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = os.path.basename(file_path)
            self.add_document(content, filename)
            print(f"✅ Loaded document from {file_path}")
            
        except Exception as e:
            print(f"❌ Error loading file {file_path}: {e}")
    
    def interactive_mode(self):
        """Run the RAG agent in interactive mode."""
        print("🤖 Simple RAG Agent")
        print("=" * 30)
        print("Commands:")
        print("  /add <text>     - Add document text")
        print("  /load <file>    - Load document from file")
        print("  /stats          - Show statistics")
        print("  /help           - Show this help")
        print("  /quit           - Exit")
        print("\nJust type a question to search for answers!")
        print()
        
        try:
            while True:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == '/quit':
                    print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == '/help':
                    print("\n📚 Available Commands:")
                    print("  /add <text>     - Add document text")
                    print("  /load <file>    - Load document from file")
                    print("  /stats          - Show statistics")
                    print("  /help           - Show this help")
                    print("  /quit           - Exit")
                    print("\nJust type a question to search for answers!")
                    
                elif user_input.lower() == '/stats':
                    stats = self.vector_store.get_stats()
                    print(f"\n📊 RAG Agent Statistics:")
                    print(f"  Documents added: {self.total_documents_added}")
                    print(f"  Total chunks: {stats['total_documents']}")
                    print(f"  Queries processed: {self.total_queries}")
                    print(f"  Embeddings generated: {self.total_embeddings_generated}")
                    print(f"  Total cost: ${self.total_cost:.4f}")
                    
                elif user_input.startswith('/add '):
                    text = user_input[5:].strip()
                    if text:
                        self.add_document(text)
                    else:
                        print("❌ Please provide text to add: /add <your text here>")
                        
                elif user_input.startswith('/load '):
                    file_path = user_input[6:].strip()
                    if file_path:
                        self.load_documents_from_file(file_path)
                    else:
                        print("❌ Please provide file path: /load <file_path>")
                        
                else:
                    # Regular question
                    if not self.vector_store.documents:
                        print("❌ No documents in knowledge base. Add some documents first!")
                        continue
                    
                    response = self.query(user_input)
                    
                    print(f"\n🤖 Answer: {response['answer']}")
                    
                    if response['sources']:
                        print(f"\n📚 Sources:")
                        for i, source in enumerate(response['sources'], 1):
                            print(f"  {i}. {source['title']} (similarity: {source['similarity']:.2f})")
                    
                    print()
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ Error: {e}")


def load_sample_documents() -> List[Dict[str, str]]:
    """Load sample documents for testing."""
    return [
        {
            "title": "Python Basics",
            "content": """Python is a high-level, interpreted programming language with dynamic semantics. 
Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it 
very attractive for Rapid Application Development, as well as for use as a scripting or glue language 
to connect existing components together. Python's simple, easy to learn syntax emphasizes readability 
and therefore reduces the cost of program maintenance. Python supports modules and packages, which 
encourages program modularity and code reuse. The Python interpreter and the extensive standard library 
are available in source or binary form without charge for all major platforms, and can be freely distributed."""
        },
        {
            "title": "Machine Learning",
            "content": """Machine learning is a method of data analysis that automates analytical model building. 
It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, 
identify patterns and make decisions with minimal human intervention. Machine learning algorithms build 
a model based on sample data, known as training data, in order to make predictions or decisions without 
being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, 
such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or 
unfeasible to develop conventional algorithms to perform the needed tasks."""
        },
        {
            "title": "RAG Systems",
            "content": """Retrieval-Augmented Generation (RAG) is a machine learning framework that combines 
the power of large language models (LLMs) with external knowledge retrieval systems. RAG addresses 
the limitation of LLMs having static knowledge by allowing them to access and incorporate up-to-date 
information from external sources during text generation. The process involves two main steps: first, 
retrieving relevant information from a knowledge base using the input query, and second, using this 
retrieved information along with the original query to generate a more informed and accurate response. 
RAG systems have shown significant improvements in factual accuracy and knowledge coverage compared 
to standalone language models."""
        }
    ]


def main():
    """Main function to demonstrate the RAG agent."""
    parser = argparse.ArgumentParser(description="Simple RAG Agent")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--load", "-l", type=str, help="Load documents from file")
    parser.add_argument("--query", "-q", type=str, help="Ask a question")
    args = parser.parse_args()
    
    print("🚀 GenAI Bootcamp - Assignment 4: Simple RAG Agent")
    print("=" * 50)
    
    try:
        # Initialize RAG agent
        agent = SimpleRAGAgent()
        
        # Load sample documents
        sample_docs = load_sample_documents()
        print("📚 Loading sample documents...")
        for doc in sample_docs:
            agent.add_document(doc["content"], doc["title"])
        
        print(f"\n✅ Loaded {len(sample_docs)} sample documents")
        
        if args.load:
            # Load additional documents from file
            agent.load_documents_from_file(args.load)
        
        if args.interactive:
            # Run interactive mode
            agent.interactive_mode()
        elif args.query:
            # Answer single question
            print(f"\n" + "="*50)
            response = agent.query(args.query)
            print(f"\n🤖 Answer: {response['answer']}")
            
            if response['sources']:
                print(f"\n📚 Sources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['title']} (similarity: {source['similarity']:.2f})")
                    
            print(f"\n📊 Confidence: {response['confidence']:.2f}")
        else:
            # Demo mode - answer some sample questions
            demo_questions = [
                "What is Python?",
                "How does machine learning work?", 
                "What is RAG and how does it work?",
                "Tell me about artificial intelligence"
            ]
            
            print(f"\n🎯 Demo Mode - Asking sample questions:")
            print("="*50)
            
            for question in demo_questions:
                print(f"\n❓ Question: {question}")
                response = agent.query(question)
                print(f"🤖 Answer: {response['answer']}")
                print(f"📊 Confidence: {response['confidence']:.2f}")
                print("-" * 30)
            
            print(f"\n💡 Try running with --interactive for full experience!")
            print(f"💡 Or use --query 'your question' for specific questions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have set up your OpenAI API key in the .env file")


if __name__ == "__main__":
    main() 