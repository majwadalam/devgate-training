#!/usr/bin/env python3
"""
Test script to verify OpenAI API connection
"""

import os
from dotenv import load_dotenv
import openai

def test_api_connection():
    """Test the OpenAI API connection."""
    print("🔍 Testing OpenAI API Connection...")
    
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
        # Make a simple test call
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello, API test successful!'"}],
            max_tokens=10
        )
        
        print("✅ API connection successful!")
        print(f"🤖 Response: {response.choices[0].message.content}")
        print(f"📊 Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

if __name__ == "__main__":
    test_api_connection() 