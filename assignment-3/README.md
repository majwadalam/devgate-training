# Assignment 3: Building Your First AI Application

## Overview
Build a simple command-line chatbot that maintains conversation memory and has a configurable personality. This assignment focuses on practical AI application development including conversation flow, memory management, and user interaction.

## Learning Objectives
- Build a complete AI application from scratch
- Implement conversation memory and context
- Create configurable system prompts and personalities
- Handle user input and AI responses
- Design simple but effective user interfaces

## Setup Instructions

### 1. Create Virtual Environment

#### On Windows:
```bash
# Navigate to assignment-3 directory
cd assignment-3

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
# Navigate to assignment-3 directory
cd assignment-3

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# For windows (Git Bash)
source venv/Scripts/activate

# For windows (Powershell)
./venv/Scripts/activate
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Set Up API Access
```bash
# Create .env file for your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

**Important:** Replace `your_api_key_here` with your actual OpenAI API key.

### 4. Verify Installation
```bash
# Test API connection
python test_api.py

# Run example chatbot
python example_script.py
```

## Assignment Requirements

### Task Description
Create a command-line chatbot that:

1. **Maintains conversation history** across multiple exchanges
2. **Has a configurable personality** through system prompts
3. **Provides a simple CLI interface** for user interaction
4. **Handles errors gracefully** and provides helpful feedback
5. **Tracks conversation statistics** (messages, tokens, cost)

### Expected Features

#### Core Features (Required):
- [ ] Command-line interface for chatting
- [ ] Conversation memory (at least last 10 messages)
- [ ] Configurable personality/system prompt
- [ ] Basic commands (/help, /clear, /stats, /quit)
- [ ] Error handling for API issues
- [ ] Cost and token tracking

#### Bonus Features (Optional):
- [ ] Save/load conversation history
- [ ] Multiple personality profiles
- [ ] Conversation export (txt/json)
- [ ] Custom commands (/joke, /explain, etc.)
- [ ] Colored terminal output

### Sample Interaction

```
🤖 Simple AI Chatbot
==================
Personality: Helpful Assistant
Type '/help' for commands, '/quit' to exit

You: Hello! How are you today?

Bot: Hello! I'm doing great, thank you for asking! I'm here and ready to help 
     with whatever you need. How can I assist you today?

You: Can you tell me a joke?

Bot: Sure! Why don't scientists trust atoms? Because they make up everything! 😄
     Is there anything specific I can help you with?

You: /stats

📊 Conversation Statistics:
- Messages exchanged: 4
- Tokens used: 156
- Estimated cost: $0.003
- Session time: 2m 15s

You: /quit

👋 Thanks for chatting! Goodbye!
```

### Expected File Structure

```
assignment-3/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── example_script.py      # Working example solution
├── assignment.py          # Starter template for students
├── test_api.py           # API connection test
├── personalities.json    # Sample personality configurations
├── .env                  # API key (create this)
└── venv/                 # Virtual environment (created by you)
```

## Getting Started

1. **Activate your virtual environment** (see setup instructions above)
2. **Set up your API key** in the `.env` file
3. **Test API connection** with `python test_api.py`
4. **Run the example script** to see expected behavior:
   ```bash
   python example_script.py
   ```
5. **Complete the assignment** by editing `assignment.py`
6. **Test your chatbot** with various conversations

## Application Architecture

### Basic Components:
1. **ChatBot Class** - Main application logic
2. **ConversationMemory** - Manages message history
3. **PersonalityManager** - Handles system prompts
4. **CLI Interface** - User interaction handling
5. **Statistics Tracker** - Usage and cost monitoring

### Key Design Patterns:
- **State Management** - Track conversation state
- **Command Pattern** - Handle special commands
- **Error Handling** - Graceful failure recovery
- **Modular Design** - Separate concerns

## Submission Guidelines

### What to Submit:
- [ ] Your completed `assignment.py` file
- [ ] Sample conversation logs or screenshots
- [ ] Any custom personality configurations you created
- [ ] Brief reflection on what you learned

### Testing Your Solution:
```bash
# Run your chatbot
python assignment.py

# Test different personalities
python assignment.py --personality friendly

# Test with specific system prompt
python assignment.py --system "You are a helpful coding assistant"
```

## Evaluation Criteria

- **Functionality (40%)** - Core features work correctly
- **User Experience (30%)** - Clean, intuitive interface
- **Code Quality (20%)** - Well-structured, readable code
- **Creativity (10%)** - Interesting personalities or features

## Common Issues & Solutions

### Memory Management:
- **Problem**: Conversation gets too long, high costs
- **Solution**: Implement message limit and trimming

### API Errors:
- **Problem**: Network timeouts or rate limits
- **Solution**: Add retry logic and user feedback

### User Input:
- **Problem**: Unexpected input crashes the app
- **Solution**: Input validation and error handling

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Command Line Interface Design](https://clig.dev/)
- [Python argparse Tutorial](https://docs.python.org/3/howto/argparse.html)
- [Conversation AI Best Practices](https://platform.openai.com/docs/guides/gpt)

## Need Help?

- Review the `example_script.py` for reference implementation
- Check the `personalities.json` for example configurations
- Ask questions during the wrap-up session
- Collaborate with peers on Discord/Slack

Good luck building your first AI application! 🚀 