#!/usr/bin/env python3
"""
Assignment 3: Building Your First AI Application Example Script
Author: GenAI Bootcamp
Description: Complete implementation of a simple command-line chatbot
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

import openai
from dotenv import load_dotenv


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_messages: int = 10):
        """Initialize conversation memory with message limit."""
        self.messages = []
        self.max_messages = max_messages
        self.start_time = datetime.now()
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        
        # Trim to max_messages, but always keep system message if it exists
        if len(self.messages) > self.max_messages:
            system_messages = [msg for msg in self.messages if msg["role"] == "system"]
            recent_messages = self.messages[-(self.max_messages-len(system_messages)):]
            self.messages = system_messages + recent_messages
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in OpenAI format."""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def clear(self) -> None:
        """Clear conversation history, keeping system message."""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages
        self.start_time = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        user_messages = len([msg for msg in self.messages if msg["role"] == "user"])
        assistant_messages = len([msg for msg in self.messages if msg["role"] == "assistant"])
        
        return {
            "total_messages": len(self.messages),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "exchanges": min(user_messages, assistant_messages),
            "session_duration": datetime.now() - self.start_time
        }


class SimpleChatbot:
    """A simple command-line chatbot with personality and memory."""
    
    def __init__(self, personality: str = "helpful_assistant", api_key: str = None):
        """Initialize the chatbot with personality and API access."""
        # Load environment variables
        load_dotenv()
        
        # Set up OpenAI client
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Load personality configuration
        self.personality_config = self.load_personality(personality)
        
        # Initialize conversation memory
        self.memory = ConversationMemory()
        
        # Add system prompt to memory
        if self.personality_config.get("system_prompt"):
            self.memory.add_message("system", self.personality_config["system_prompt"])
        
        # Statistics tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        self.session_start = datetime.now()
        
        # Cost per 1K tokens (gpt-4o-mini)
        self.cost_per_1k_input = 0.0015
        self.cost_per_1k_output = 0.006
        
        print(f"✅ Chatbot initialized with personality: {self.personality_config['name']}")
    
    def load_personality(self, personality_name: str) -> Dict[str, str]:
        """Load personality configuration from JSON file."""
        try:
            with open("personalities.json", "r") as f:
                personalities = json.load(f)
            
            if personality_name in personalities:
                return personalities[personality_name]
            else:
                print(f"⚠️  Personality '{personality_name}' not found, using helpful_assistant")
                return personalities.get("helpful_assistant", {
                    "name": "Default Assistant",
                    "system_prompt": "You are a helpful assistant.",
                    "description": "Default helpful assistant"
                })
        except FileNotFoundError:
            print("⚠️  personalities.json not found, using default personality")
            return {
                "name": "Default Assistant",
                "system_prompt": "You are a helpful assistant.",
                "description": "Default helpful assistant"
            }
    
    def call_openai_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make a call to OpenAI API with error handling."""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens / 1000 * self.cost_per_1k_input + 
                   output_tokens / 1000 * self.cost_per_1k_output)
            
            # Update statistics
            self.total_tokens += response.usage.total_tokens
            self.total_cost += cost
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "cost": cost
            }
            
        except openai.error.RateLimitError:
            return {"error": "Rate limit exceeded. Please wait a moment and try again."}
        except openai.error.APIError as e:
            return {"error": f"API error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def process_command(self, command: str) -> bool:
        """Process special commands that start with '/'."""
        command = command.lower().strip()
        
        if command == "/help":
            print("\n📚 Available Commands:")
            print("  /help    - Show this help message")
            print("  /clear   - Clear conversation history")
            print("  /stats   - Show conversation statistics")
            print("  /quit    - Exit the chatbot")
            print("  Just type normally to chat!\n")
            return True
            
        elif command == "/clear":
            self.memory.clear()
            print("🧹 Conversation history cleared!\n")
            return True
            
        elif command == "/stats":
            stats = self.get_session_stats()
            print(f"\n📊 Conversation Statistics:")
            print(f"  Messages exchanged: {stats['exchanges']}")
            print(f"  Total tokens used: {stats['total_tokens']}")
            print(f"  Estimated cost: ${stats['total_cost']:.4f}")
            print(f"  Session time: {stats['session_duration']}")
            print()
            return True
            
        elif command == "/quit":
            print("\n👋 Thanks for chatting! Goodbye!")
            return False
            
        else:
            print(f"❓ Unknown command: {command}")
            print("Type '/help' to see available commands.\n")
            return True
    
    def chat_loop(self) -> None:
        """Main chat interaction loop."""
        print(f"\n🤖 Simple AI Chatbot")
        print("=" * 30)
        print(f"Personality: {self.personality_config['name']}")
        print(f"Description: {self.personality_config.get('description', 'No description')}")
        print("Type '/help' for commands, '/quit' to exit")
        print()
        
        try:
            while True:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self.process_command(user_input):
                        break  # /quit was entered
                    continue
                
                # Add user message to memory
                self.memory.add_message("user", user_input)
                
                # Get AI response
                print("Bot: ", end="", flush=True)
                
                messages = self.memory.get_messages()
                api_response = self.call_openai_api(messages)
                
                if "error" in api_response:
                    print(f"❌ {api_response['error']}")
                    # Remove the user message since we couldn't respond
                    self.memory.messages.pop()
                else:
                    # Display and store AI response
                    response_text = api_response["content"]
                    print(response_text)
                    self.memory.add_message("assistant", response_text)
                
                print()  # Add spacing
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        memory_stats = self.memory.get_stats()
        session_duration = datetime.now() - self.session_start
        
        # Format duration nicely
        total_seconds = int(session_duration.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        duration_str = f"{minutes}m {seconds}s"
        
        return {
            "exchanges": memory_stats["exchanges"],
            "total_messages": memory_stats["total_messages"],
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "session_duration": duration_str
        }


def load_personalities() -> Dict[str, Any]:
    """Load available personalities from JSON file."""
    try:
        with open("personalities.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  personalities.json not found")
        return {}


def main():
    """Main function to run the chatbot."""
    parser = argparse.ArgumentParser(description="Simple AI Chatbot")
    parser.add_argument("--personality", "-p", type=str, default="helpful_assistant",
                       help="Personality to use (see personalities.json)")
    parser.add_argument("--system", "-s", type=str, help="Custom system prompt")
    parser.add_argument("--list-personalities", "-l", action="store_true",
                       help="List available personalities")
    args = parser.parse_args()
    
    print("🚀 GenAI Bootcamp - Assignment 3: Building Your First AI Application")
    print("=" * 60)
    
    # List personalities if requested
    if args.list_personalities:
        personalities = load_personalities()
        if personalities:
            print("\n📝 Available Personalities:")
            for key, config in personalities.items():
                print(f"  {key}: {config.get('description', 'No description')}")
        else:
            print("\n❌ No personalities found. Make sure personalities.json exists.")
        return
    
    try:
        # Create and run chatbot
        chatbot = SimpleChatbot(personality=args.personality)
        
        # Override system prompt if provided
        if args.system:
            chatbot.personality_config["system_prompt"] = args.system
            chatbot.memory.clear()  # Clear and re-add system message
            chatbot.memory.add_message("system", args.system)
            print(f"✅ Using custom system prompt")
        
        chatbot.chat_loop()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have set up your OpenAI API key in the .env file")


if __name__ == "__main__":
    main() 