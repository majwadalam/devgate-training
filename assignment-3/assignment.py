#!/usr/bin/env python3
"""
Assignment 3: Building Your First AI Application - Student Template
Name: [Your Name Here]
Date: [Date]
Description: Complete this script to build a simple command-line chatbot
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

# TODO: Import required libraries
# import openai
# from dotenv import load_dotenv


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_messages: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_messages (int): Maximum number of messages to keep in memory
            
        TODO: Implement initialization
        - Set up message storage
        - Set maximum message limit
        """
        # TODO: Add your initialization code here
        pass
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role (str): Either 'user' or 'assistant'
            content (str): Message content
            
        TODO: Implement message addition
        - Add message to history
        - Trim history if it exceeds max_messages
        """
        # TODO: Your code here
        pass
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in conversation history.
        
        Returns:
            List[Dict]: List of message dictionaries
            
        TODO: Return the conversation history
        """
        # TODO: Your code here
        pass
    
    def clear(self) -> None:
        """
        Clear conversation history.
        
        TODO: Clear all stored messages
        """
        # TODO: Your code here
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dict: Statistics about the conversation
            
        TODO: Calculate and return stats like message count
        """
        # TODO: Your code here
        pass


class SimpleChatbot:
    """A simple command-line chatbot with personality and memory."""
    
    def __init__(self, personality: str = "helpful_assistant", api_key: str = None):
        """
        Initialize the chatbot.
        
        Args:
            personality (str): Personality configuration to use
            api_key (str): OpenAI API key (optional, can use environment variable)
            
        TODO: Implement initialization
        - Load environment variables
        - Set up OpenAI client
        - Load personality configuration
        - Initialize conversation memory
        - Set up statistics tracking
        """
        # TODO: Add your initialization code here
        pass
    
    def load_personality(self, personality_name: str) -> Dict[str, str]:
        """
        Load personality configuration from JSON file.
        
        Args:
            personality_name (str): Name of personality to load
            
        Returns:
            Dict: Personality configuration
            
        TODO: Load personality from personalities.json
        - Read the JSON file
        - Return the specified personality
        - Handle file not found errors
        """
        # TODO: Your code here
        pass
    
    def call_openai_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Make a call to OpenAI API.
        
        Args:
            messages (List[Dict]): Conversation messages
            
        Returns:
            Dict: API response with content and usage info
            
        TODO: Implement API call
        - Make request to OpenAI
        - Handle errors gracefully
        - Track token usage and costs
        - Return response data
        """
        # TODO: Your code here
        pass
    
    def process_command(self, command: str) -> bool:
        """
        Process special commands that start with '/'.
        
        Args:
            command (str): Command to process
            
        Returns:
            bool: True if command was processed, False to continue chat
            
        TODO: Implement command processing
        - Handle /help, /clear, /stats, /quit commands
        - Return appropriate responses
        - Return False for /quit to exit
        """
        # TODO: Your code here
        pass
    
    def chat_loop(self) -> None:
        """
        Main chat interaction loop.
        
        TODO: Implement the main chat interface
        - Display welcome message
        - Get user input in a loop
        - Process commands or send to AI
        - Display responses
        - Handle KeyboardInterrupt (Ctrl+C)
        """
        print(f"🤖 Simple AI Chatbot")
        print("=" * 30)
        print(f"Personality: {self.personality_config.get('name', 'Unknown')}")
        print("Type '/help' for commands, '/quit' to exit")
        print()
        
        # TODO: Implement the chat loop
        # Hint: Use while True loop with input()
        # Hint: Check if input starts with '/' for commands
        # Hint: Handle exceptions gracefully
        
        pass
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current session.
        
        Returns:
            Dict: Session statistics
            
        TODO: Calculate session statistics
        - Total messages exchanged
        - Total tokens used
        - Total cost
        - Session duration
        """
        # TODO: Your code here
        pass


def load_personalities() -> Dict[str, Any]:
    """Load available personalities from JSON file."""
    try:
        with open("personalities.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  personalities.json not found, using default personality")
        return {
            "default": {
                "name": "Default Assistant",
                "system_prompt": "You are a helpful assistant.",
                "description": "Default helpful assistant"
            }
        }


def main():
    """Main function to run the chatbot."""
    parser = argparse.ArgumentParser(description="Simple AI Chatbot")
    parser.add_argument("--personality", "-p", type=str, default="helpful_assistant",
                       help="Personality to use (see personalities.json)")
    parser.add_argument("--system", "-s", type=str, help="Custom system prompt")
    parser.add_argument("--list-personalities", "-l", action="store_true",
                       help="List available personalities")
    args = parser.parse_args()
    
    print("🚀 Assignment 3: Building Your First AI Application")
    print("=" * 50)
    
    # List personalities if requested
    if args.list_personalities:
        personalities = load_personalities()
        print("\n📝 Available Personalities:")
        for key, config in personalities.items():
            print(f"  {key}: {config.get('description', 'No description')}")
        return
    
    try:
        # TODO: Create and run chatbot
        # chatbot = SimpleChatbot(personality=args.personality)
        # if args.system:
        #     chatbot.personality_config["system_prompt"] = args.system
        # chatbot.chat_loop()
        
        print("💡 Complete the TODO sections to make the chatbot work!")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have set up your OpenAI API key in the .env file")


# Test your implementation
if __name__ == "__main__":
    main()


# BONUS CHALLENGES (Optional):
# 1. Add colored terminal output using colorama
# 2. Implement conversation saving/loading
# 3. Add custom commands like /joke, /explain, /translate
# 4. Create a typing animation effect
# 5. Add conversation export to text/JSON
# 6. Implement conversation search functionality


# TESTING CHECKLIST:
# □ Script runs without errors
# □ Can have basic conversations
# □ Commands work (/help, /clear, /stats, /quit)
# □ Conversation memory works
# □ Personality system works
# □ API errors handled gracefully
# □ Statistics tracking works
# □ Code is well-commented and readable


# SUBMISSION:
# 1. Complete all TODO sections
# 2. Test with different personalities
# 3. Have a few sample conversations
# 4. Add your name and date at the top
# 5. Submit assignment.py and any conversation logs
# 6. Include a brief reflection on what you learned 