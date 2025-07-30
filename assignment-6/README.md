# Assignment 6: Advanced RAG Techniques

## Overview
Optimize RAG system performance and create evaluation benchmarks. This assignment focuses on advanced chunking strategies, metadata filtering, hybrid search, and systematic evaluation of RAG performance.

## Learning Objectives
- Implement different chunking strategies and compare their effectiveness
- Add metadata filtering to improve retrieval accuracy
- Create evaluation metrics for RAG systems
- Optimize RAG performance through systematic testing
- Understand hybrid search combining keyword and semantic search

## Setup Instructions

### 1. Create Virtual Environment
```bash
# Navigate to assignment-6 directory
cd assignment-6

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# OR
venv\Scripts\activate  # Windows CMD/PowerShell
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Assignment Requirements

### Task Description
Enhance your RAG system with:

1. **Multiple Chunking Strategies** - implement and compare different text splitting approaches
2. **Metadata Filtering** - add document metadata for better retrieval
3. **Hybrid Search** - combine keyword and semantic search
4. **Evaluation Framework** - create metrics to measure RAG performance
5. **Performance Optimization** - systematic testing and improvement

### Expected Features

#### Core Features (Required):
- [ ] Implement 3 different chunking strategies
- [ ] Add metadata filtering to documents
- [ ] Create evaluation metrics (relevance, accuracy, latency)
- [ ] Compare performance across different configurations
- [ ] Generate performance benchmarks and reports

#### Bonus Features (Optional):
- [ ] Implement hybrid search (BM25 + vector search)
- [ ] Add query expansion techniques
- [ ] Create A/B testing framework
- [ ] Implement result re-ranking
- [ ] Add caching for improved performance

### Chunking Strategies to Implement

#### 1. Fixed-Size Chunking:
- Simple character-based splitting
- Fixed chunk size with overlap

#### 2. Semantic Chunking:
- Split by sentences or paragraphs
- Preserve meaning boundaries

#### 3. Hierarchical Chunking:
- Create chunks at multiple levels
- Parent-child relationships

### Evaluation Metrics

#### Retrieval Metrics:
- **Relevance Score**: How relevant retrieved chunks are to the query
- **Recall@K**: How many relevant documents are in top K results
- **Precision@K**: What fraction of top K results are relevant

#### Response Quality:
- **Answer Accuracy**: Correctness of generated responses
- **Faithfulness**: How well the answer reflects the source material
- **Completeness**: Whether the answer fully addresses the question

#### Performance Metrics:
- **Latency**: Time to generate responses
- **Throughput**: Queries processed per second
- **Token Usage**: API costs and efficiency

## Files Structure

```
assignment-6/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── assignment.py          # Your solution template
├── example_optimized.py   # Working example
├── evaluation.py          # Evaluation framework
├── test_data/            # Test questions and expected answers
│   ├── test_questions.json
│   └── ground_truth.json
├── documents/            # Sample documents with metadata
└── results/              # Generated benchmark reports
```

## Getting Started

1. **Run the baseline** to see current performance:
   ```bash
   python example_optimized.py --baseline
   ```

2. **Implement your optimizations** in `assignment.py`

3. **Run evaluations** to measure improvements:
   ```bash
   python evaluation.py --config your_config.json
   ```

4. **Generate benchmark report**:
   ```bash
   python assignment.py --benchmark
   ```

## Key Concepts

### 1. Chunking Strategy Comparison
```python
strategies = {
    'fixed': {'chunk_size': 1000, 'overlap': 200},
    'semantic': {'method': 'sentence', 'max_size': 1000},
    'hierarchical': {'levels': 2, 'sizes': [2000, 500]}
}
```

### 2. Metadata Integration
```python
metadata = {
    'source': 'filename.txt',
    'section': 'Introduction',
    'topic': 'AI Basics',
    'difficulty': 'beginner',
    'last_updated': '2024-01-01'
}
```

### 3. Evaluation Framework
```python
evaluation_results = {
    'retrieval_metrics': {'recall@5': 0.85, 'precision@5': 0.75},
    'response_quality': {'accuracy': 0.82, 'faithfulness': 0.90},
    'performance': {'avg_latency': 1.2, 'tokens_used': 1500}
}
```

## Sample Test Questions

Create comprehensive test questions covering:
- Factual questions (What is...?)
- Comparative questions (Compare X and Y)
- Multi-hop reasoning (Questions requiring multiple documents)
- Edge cases (Ambiguous or out-of-domain questions)

## Submission Guidelines

### What to Submit:
- [ ] Completed `assignment.py` with all optimizations
- [ ] Evaluation results comparing different strategies
- [ ] Performance benchmark report
- [ ] Analysis of findings and recommendations
- [ ] Any custom test questions you created

### Expected Deliverables:
1. **Performance Report**: Detailed comparison of chunking strategies
2. **Optimization Analysis**: What improvements worked and why
3. **Benchmark Results**: Quantitative metrics for your system
4. **Recommendations**: Best practices for RAG optimization

## Advanced Techniques (Bonus)

### Hybrid Search Implementation:
```python
# Combine BM25 and vector search
bm25_results = bm25_retriever.get_relevant_documents(query)
vector_results = vector_retriever.get_relevant_documents(query)
combined_results = combine_and_rerank(bm25_results, vector_results)
```

### Query Expansion:
```python
# Expand queries with synonyms and related terms
expanded_query = query_expander.expand(original_query)
```

## Common Optimization Patterns

1. **Chunk Size Tuning**: Test different sizes for your domain
2. **Overlap Optimization**: Find the right balance for context preservation
3. **Retrieval Count**: Optimize number of chunks to retrieve
4. **Re-ranking**: Post-process results for better relevance
5. **Caching**: Cache embeddings and frequent queries

## Need Help?

- Start with the example implementation
- Use the evaluation framework to measure improvements
- Test one optimization at a time
- Document your experiments and findings

Ready to optimize your RAG system! 🚀📊 