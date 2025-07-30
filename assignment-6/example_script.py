"""
Assignment 6: Example Optimized RAG with Evaluation
This demonstrates advanced RAG techniques including multiple chunking strategies,
evaluation metrics, and performance optimization.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

load_dotenv()

@dataclass
class ChunkingConfig:
    """Configuration for different chunking strategies"""
    strategy: str
    chunk_size: int
    overlap: int
    additional_params: Dict[str, Any] = None

@dataclass
class EvaluationResult:
    """Results from evaluating RAG performance"""
    config: ChunkingConfig
    retrieval_metrics: Dict[str, float]
    response_quality: Dict[str, float]
    performance_metrics: Dict[str, float]
    test_results: List[Dict[str, Any]]

class OptimizedRAG:
    """Advanced RAG system with multiple strategies and evaluation"""
    
    def __init__(self, documents_path: str = "documents"):
        """Initialize the optimized RAG system"""
        self.documents_path = documents_path
        
        # Initialize OpenAI components
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.embeddings = OpenAIEmbeddings()
        
        # Current configuration
        self.current_config = None
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.bm25 = None
        self.documents = []
        
        # Load base documents
        self._load_base_documents()
    
    def _load_base_documents(self):
        """Load base documents that will be chunked differently"""
        if not os.path.exists(self.documents_path):
            print(f"Documents directory '{self.documents_path}' not found!")
            return
        
        for filename in os.listdir(self.documents_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.documents_path, filename)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    self.documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def apply_chunking_strategy(self, config: ChunkingConfig) -> List[Document]:
        """Apply different chunking strategies"""
        self.current_config = config
        
        if config.strategy == "fixed":
            return self._fixed_chunking(config)
        elif config.strategy == "semantic":
            return self._semantic_chunking(config)
        elif config.strategy == "hierarchical":
            return self._hierarchical_chunking(config)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
    
    def _fixed_chunking(self, config: ChunkingConfig) -> List[Document]:
        """Fixed-size chunking strategy"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap
        )
        return splitter.split_documents(self.documents)
    
    def _semantic_chunking(self, config: ChunkingConfig) -> List[Document]:
        """Semantic chunking by sentences/paragraphs"""
        chunks = []
        for doc in self.documents:
            # Split by double newlines (paragraphs)
            paragraphs = doc.page_content.split('\n\n')
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk + paragraph) <= config.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={**doc.metadata, "chunk_type": "semantic"}
                        ))
                    current_chunk = paragraph + "\n\n"
            
            if current_chunk:
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={**doc.metadata, "chunk_type": "semantic"}
                ))
        
        return chunks
    
    def _hierarchical_chunking(self, config: ChunkingConfig) -> List[Document]:
        """Hierarchical chunking with parent-child relationships"""
        # Create large parent chunks
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size * 2,
            chunk_overlap=config.overlap
        )
        parent_chunks = parent_splitter.split_documents(self.documents)
        
        # Create smaller child chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size // 2,
            chunk_overlap=config.overlap
        )
        
        all_chunks = []
        for i, parent in enumerate(parent_chunks):
            # Add parent chunk
            parent.metadata.update({
                "chunk_type": "parent",
                "parent_id": i
            })
            all_chunks.append(parent)
            
            # Add child chunks
            children = child_splitter.split_documents([parent])
            for j, child in enumerate(children):
                child.metadata.update({
                    "chunk_type": "child",
                    "parent_id": i,
                    "child_id": j
                })
                all_chunks.append(child)
        
        return all_chunks
    
    def setup_vectorstore(self, chunks: List[Document]):
        """Set up vector store with chunks"""
        if not chunks:
            return
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=f"./chroma_db_{self.current_config.strategy}"
        )
        
        # Set up retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Set up RAG chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        
        # Set up BM25 for hybrid search
        self._setup_bm25(chunks)
    
    def _setup_bm25(self, chunks: List[Document]):
        """Set up BM25 for hybrid search"""
        try:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            
            # Tokenize documents
            tokenized_docs = []
            for chunk in chunks:
                tokens = word_tokenize(chunk.page_content.lower())
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
                tokenized_docs.append(tokens)
            
            self.bm25 = BM25Okapi(tokenized_docs)
            self.bm25_chunks = chunks
            
        except Exception as e:
            print(f"Warning: Could not set up BM25: {e}")
            self.bm25 = None
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Combine BM25 and vector search"""
        if not self.bm25 or not self.retriever:
            return self.retriever.get_relevant_documents(query) if self.retriever else []
        
        try:
            from nltk.tokenize import word_tokenize
            
            # BM25 search
            query_tokens = word_tokenize(query.lower())
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:k*2]
            
            # Vector search
            vector_docs = self.retriever.get_relevant_documents(query)
            
            # Combine results (simple approach)
            combined_docs = []
            
            # Add top BM25 results
            for idx in bm25_top_indices[:k//2]:
                combined_docs.append(self.bm25_chunks[idx])
            
            # Add vector search results
            for doc in vector_docs[:k//2]:
                if doc not in combined_docs:
                    combined_docs.append(doc)
            
            return combined_docs[:k]
            
        except Exception as e:
            print(f"Hybrid search failed, falling back to vector search: {e}")
            return self.retriever.get_relevant_documents(query)
    
    def evaluate_retrieval(self, question: str, retrieved_docs: List[Document], 
                          relevant_keywords: List[str]) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        if not retrieved_docs:
            return {"precision_at_5": 0.0, "recall_at_5": 0.0, "f1_at_5": 0.0}
        
        # Simple keyword-based relevance (in real scenario, use human annotations)
        relevant_count = 0
        for doc in retrieved_docs:
            doc_text = doc.page_content.lower()
            if any(keyword.lower() in doc_text for keyword in relevant_keywords):
                relevant_count += 1
        
        precision = relevant_count / len(retrieved_docs) if retrieved_docs else 0
        # For recall, we'd need ground truth - using precision as proxy
        recall = precision  # Simplified
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision_at_5": precision,
            "recall_at_5": recall,
            "f1_at_5": f1
        }
    
    def evaluate_response_quality(self, question: str, answer: str) -> Dict[str, float]:
        """Evaluate response quality"""
        # Simple heuristic-based evaluation
        metrics = {
            "length_score": min(len(answer) / 100, 1.0),  # Prefer reasonable length
            "completeness": 1.0 if len(answer) > 50 else 0.5,  # Basic completeness check
            "relevance": 0.8  # Would need more sophisticated evaluation
        }
        
        return metrics
    
    def run_evaluation(self, config: ChunkingConfig, test_questions: List[Dict]) -> EvaluationResult:
        """Run comprehensive evaluation for a configuration"""
        print(f"Evaluating config: {config.strategy} (chunk_size={config.chunk_size})")
        
        # Apply chunking strategy
        chunks = self.apply_chunking_strategy(config)
        self.setup_vectorstore(chunks)
        
        if not self.chain:
            print("Failed to set up RAG chain")
            return None
        
        # Run evaluation on test questions
        test_results = []
        retrieval_scores = []
        quality_scores = []
        performance_scores = []
        
        for test_case in tqdm(test_questions, desc="Testing questions"):
            question = test_case["question"]
            keywords = test_case.get("keywords", [])
            
            # Measure performance
            start_time = time.time()
            
            try:
                # Get answer
                result = self.chain.invoke({"query": question})
                answer = result.get("result", "")
                retrieved_docs = result.get("source_documents", [])
                
                end_time = time.time()
                
                # Evaluate retrieval
                retrieval_metrics = self.evaluate_retrieval(question, retrieved_docs, keywords)
                retrieval_scores.append(retrieval_metrics)
                
                # Evaluate response quality
                quality_metrics = self.evaluate_response_quality(question, answer)
                quality_scores.append(quality_metrics)
                
                # Performance metrics
                performance_metrics = {
                    "latency": end_time - start_time,
                    "retrieved_docs": len(retrieved_docs)
                }
                performance_scores.append(performance_metrics)
                
                # Store individual result
                test_results.append({
                    "question": question,
                    "answer": answer,
                    "retrieval_metrics": retrieval_metrics,
                    "quality_metrics": quality_metrics,
                    "performance_metrics": performance_metrics
                })
                
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                continue
        
        # Aggregate metrics
        avg_retrieval = {}
        avg_quality = {}
        avg_performance = {}
        
        if retrieval_scores:
            for key in retrieval_scores[0].keys():
                avg_retrieval[key] = np.mean([score[key] for score in retrieval_scores])
        
        if quality_scores:
            for key in quality_scores[0].keys():
                avg_quality[key] = np.mean([score[key] for score in quality_scores])
        
        if performance_scores:
            for key in performance_scores[0].keys():
                avg_performance[key] = np.mean([score[key] for score in performance_scores])
        
        return EvaluationResult(
            config=config,
            retrieval_metrics=avg_retrieval,
            response_quality=avg_quality,
            performance_metrics=avg_performance,
            test_results=test_results
        )

def create_test_questions() -> List[Dict]:
    """Create test questions with relevant keywords"""
    return [
        {
            "question": "What is artificial intelligence?",
            "keywords": ["artificial", "intelligence", "AI", "machine", "computer"]
        },
        {
            "question": "How is Python used in machine learning?",
            "keywords": ["Python", "machine learning", "programming", "libraries"]
        },
        {
            "question": "What are the types of machine learning?",
            "keywords": ["supervised", "unsupervised", "reinforcement", "learning"]
        },
        {
            "question": "What are neural networks?",
            "keywords": ["neural", "networks", "deep learning", "neurons"]
        },
        {
            "question": "What are the challenges in AI development?",
            "keywords": ["challenges", "AI", "development", "problems", "issues"]
        }
    ]

def main():
    """Main function to run optimization evaluation"""
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        return
    
    # Initialize system
    rag = OptimizedRAG()
    
    if not rag.documents:
        print("❌ No documents found. Please add .txt files to documents/ directory")
        return
    
    # Test configurations
    configs = [
        ChunkingConfig("fixed", 500, 100),
        ChunkingConfig("fixed", 1000, 200),
        ChunkingConfig("fixed", 1500, 300),
        ChunkingConfig("semantic", 1000, 0),
        ChunkingConfig("hierarchical", 1000, 200)
    ]
    
    # Test questions
    test_questions = create_test_questions()
    
    print(f"🧪 Starting evaluation with {len(configs)} configurations")
    print(f"📝 Testing with {len(test_questions)} questions")
    print("="*60)
    
    # Run evaluations
    results = []
    for config in configs:
        try:
            result = rag.run_evaluation(config, test_questions)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {config.strategy}: {e}")
    
    # Generate report
    if results:
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        for result in results:
            row = {
                "Strategy": result.config.strategy,
                "Chunk Size": result.config.chunk_size,
                "Overlap": result.config.overlap,
                "Precision@5": f"{result.retrieval_metrics.get('precision_at_5', 0):.3f}",
                "F1@5": f"{result.retrieval_metrics.get('f1_at_5', 0):.3f}",
                "Avg Latency": f"{result.performance_metrics.get('latency', 0):.2f}s",
                "Completeness": f"{result.response_quality.get('completeness', 0):.3f}"
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Find best configuration
        best_result = max(results, key=lambda r: r.retrieval_metrics.get('f1_at_5', 0))
        print(f"\n🏆 Best Configuration:")
        print(f"   Strategy: {best_result.config.strategy}")
        print(f"   Chunk Size: {best_result.config.chunk_size}")
        print(f"   F1 Score: {best_result.retrieval_metrics.get('f1_at_5', 0):.3f}")
        
        # Save results
        os.makedirs("results", exist_ok=True)
        
        # Save detailed results
        with open("results/evaluation_results.json", "w") as f:
            results_data = []
            for result in results:
                results_data.append({
                    "config": asdict(result.config),
                    "metrics": {
                        "retrieval": result.retrieval_metrics,
                        "quality": result.response_quality,
                        "performance": result.performance_metrics
                    }
                })
            json.dump(results_data, f, indent=2)
        
        # Save comparison table
        df.to_csv("results/comparison_table.csv", index=False)
        
        print(f"\n💾 Results saved to results/ directory")
    
    else:
        print("❌ No successful evaluations completed")

if __name__ == "__main__":
    main() 