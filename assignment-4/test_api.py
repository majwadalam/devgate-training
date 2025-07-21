#!/usr/bin/env python3
"""
Test script to verify OpenAI API connection and embeddings functionality
"""

import os
from dotenv import load_dotenv
import openai

def test_api_connection():
    """Test the OpenAI API connection and embeddings."""
    print("🔍 Testing OpenAI API Connection for RAG...")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("💡 Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Set up OpenAI client
    openai.api_key = api_key
    
    try:
        # Test embeddings API
        print("🧮 Testing embeddings API...")
        embedding_response = openai.embeddings.create(
            model="text-embedding-3-small",
            input="This is a test document for RAG system."
        )
        
        embedding = embedding_response.data[0].embedding
        print(f"✅ Embeddings API works! Vector dimension: {len(embedding)}")
        
        # Test chat completion API
        print("💬 Testing chat completion API...")
        chat_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'RAG test successful!'"}],
            max_tokens=10
        )
        
        print(f"✅ Chat API works! Response: {chat_response.choices[0].message.content}")
        
        # Test basic similarity calculation
        print("📏 Testing similarity calculation...")
        # Create two similar embeddings
        doc1_response = openai.embeddings.create(
            model="text-embedding-3-small",
            input="Python is a programming language"
        )
        doc2_response = openai.embeddings.create(
            model="text-embedding-3-small", 
            input="Python is used for coding"
        )
        
        # Simple cosine similarity calculation
        import math
        vec1 = doc1_response.data[0].embedding
        vec2 = doc2_response.data[0].embedding
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        similarity = dot_product / (magnitude1 * magnitude2)
        
        print(f"✅ Similarity calculation works! Similarity: {similarity:.4f}")
        
        print("\n🎉 All tests passed! Your RAG system is ready to go!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    test_api_connection() 