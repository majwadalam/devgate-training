# Assignment 2: Prompt Engineering Playground

## Overview
Build a comprehensive prompt engineering playground that tests and compares different prompting techniques. This assignment focuses on understanding how to effectively communicate with Large Language Models (LLMs) through various prompting strategies and API integration.

## Learning Objectives
- Set up and configure OpenAI API access
- Implement different prompting techniques (zero-shot, few-shot, chain-of-thought)
- Compare and evaluate prompt effectiveness
- Handle API rate limiting and error management
- Create a systematic approach to prompt testing

## Setup Instructions

### 1. Create Virtual Environment

#### On Windows:
```bash
# Navigate to assignment-2 directory
cd assignment-2

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
# Navigate to assignment-2 directory
cd assignment-2

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

**Important:** Replace `your_api_key_here` with your actual OpenAI API key. Never commit your API key to version control!

### 4. Verify Installation
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Test API connection (after setting up .env)
python test_api.py
```

## Assignment Requirements

### Task Description
Create a Python application that:

1. **Tests different prompting techniques** (zero-shot, few-shot, chain-of-thought)
2. **Compares prompt effectiveness** across various tasks
3. **Manages API calls efficiently** with rate limiting
4. **Generates detailed reports** on prompt performance
5. **Provides an interactive interface** for prompt experimentation

### Expected Features

#### Core Features (Required):
- [ ] OpenAI API integration with proper error handling
- [ ] Zero-shot prompting implementation
- [ ] Few-shot prompting with examples
- [ ] Chain-of-thought reasoning prompts
- [ ] Response comparison and evaluation
- [ ] Rate limiting and cost tracking

#### Bonus Features (Optional):
- [ ] Web interface using Streamlit or Flask
- [ ] Prompt template management system
- [ ] A/B testing framework for prompts
- [ ] Response quality scoring
- [ ] Export results to CSV/JSON
- [ ] Custom task definitions

### Sample Tasks to Test

#### 1. Text Classification
```python
task = "Classify the sentiment of this text: 'I love this product, it's amazing!'"
expected = "positive"
```

#### 2. Mathematical Reasoning
```python
task = "If a train travels 120 km in 2 hours, what is its speed in km/h?"
expected = "60 km/h"
```

#### 3. Creative Writing
```python
task = "Write a short story about a robot learning to paint"
expected = "Creative narrative response"
```

#### 4. Code Generation
```python
task = "Write a Python function to calculate fibonacci numbers"
expected = "Working Python code"
```

### Expected Output Format

#### Console Output:
```
🤖 Prompt Engineering Playground
================================

Testing Zero-Shot Prompting...
✅ Task 1: Sentiment Analysis
   Response: positive
   Confidence: 95%
   Cost: $0.002

Testing Few-Shot Prompting...
✅ Task 1: Sentiment Analysis  
   Response: positive
   Confidence: 98%
   Cost: $0.003

Testing Chain-of-Thought...
✅ Task 2: Math Problem
   Response: 60 km/h
   Reasoning: If distance = 120km and time = 2h, then speed = 120/2 = 60 km/h
   Cost: $0.005

📊 Summary Report:
- Total API calls: 9
- Total cost: $0.015
- Best performing technique: Few-shot (98% accuracy)
- Average response time: 2.3s
```

#### JSON Report:
```json
{
  "session_info": {
    "timestamp": "2024-01-15T10:30:00Z",
    "total_tasks": 4,
    "total_cost": 0.015,
    "api_calls": 9
  },
  "techniques": {
    "zero_shot": {
      "accuracy": 0.75,
      "avg_cost": 0.002,
      "avg_time": 2.1
    },
    "few_shot": {
      "accuracy": 0.98,
      "avg_cost": 0.003,
      "avg_time": 2.5
    },
    "chain_of_thought": {
      "accuracy": 0.92,
      "avg_cost": 0.005,
      "avg_time": 3.2
    }
  },
  "recommendations": [
    "Use few-shot prompting for classification tasks",
    "Chain-of-thought works best for reasoning problems",
    "Consider cost vs accuracy trade-offs"
  ]
}
```

## Files Structure

```
assignment-2/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── example_script.py      # Working example solution
├── assignment.py          # Starter template for students
├── test_api.py           # API connection test
├── sample_tasks.json     # Sample tasks for testing
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
6. **Test your solution** with different prompts and tasks

## Prompt Engineering Techniques

### 1. Zero-Shot Prompting
Direct questions without examples:
```
"Classify the sentiment: 'I love this movie!'"
```

### 2. Few-Shot Prompting
Provide examples before the task:
```
"Sentiment: 'I hate this' → negative
Sentiment: 'This is great' → positive  
Sentiment: 'I love this movie!' → ?"
```

### 3. Chain-of-Thought (CoT)
Encourage step-by-step reasoning:
```
"Let's solve this step by step:
If a train travels 120 km in 2 hours, what is its speed?
Step 1: We know distance = 120 km and time = 2 hours
Step 2: Speed = distance ÷ time
Step 3: Speed = 120 ÷ 2 = 60 km/h
Therefore, the speed is 60 km/h"
```

### 4. System Prompts
Set the AI's role and behavior:
```
"You are a helpful math tutor. Always show your work step by step."
```

## Submission Guidelines

### What to Submit:
- [ ] Your completed `assignment.py` file
- [ ] Generated reports (JSON/CSV files)
- [ ] Screenshots of your playground in action
- [ ] Brief documentation of your findings

### Testing Your Solution:
```bash
# Run your assignment script
python assignment.py

# Test with custom prompts
python assignment.py --task "custom_task_here"

# Generate detailed report
python assignment.py --report detailed
```

## Common Issues & Solutions

### API Key Issues:
- **Problem**: "Invalid API key" error
- **Solution**: Check your `.env` file and API key validity

### Rate Limiting:
- **Problem**: "Rate limit exceeded" errors
- **Solution**: Implement exponential backoff and request delays

### Cost Management:
- **Problem**: Unexpected high costs
- **Solution**: Set usage limits and monitor token usage

### Response Quality:
- **Problem**: Inconsistent or poor responses
- **Solution**: Refine prompts and add more context/examples

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Rate Limiting Best Practices](https://platform.openai.com/docs/guides/rate-limits)

## Need Help?

- Review the `example_script.py` for reference implementation
- Check the `sample_tasks.json` for task examples
- Ask questions during the wrap-up session
- Collaborate with peers on Discord/Slack

Good luck with your prompt engineering assignment! 🚀 