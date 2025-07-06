#!/usr/bin/env python3
"""
Assignment 2: Prompt Engineering Playground - Student Template
Name: [Your Name Here]
Date: [Date]
Description: Complete this script to test different prompting techniques with OpenAI API
"""

import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# TODO: Import required libraries
# import openai
# from dotenv import load_dotenv


@dataclass
class TaskResult:
    """Data class to store task execution results."""
    task_id: str
    technique: str
    prompt: str
    response: str
    expected: str
    cost: float
    response_time: float
    accuracy: float
    confidence: float


class PromptEngineer:
    """Complete this class to test different prompting techniques."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the PromptEngineer with OpenAI API access.
        
        Args:
            api_key (str): OpenAI API key (optional, can use environment variable)
            
        TODO: Implement initialization
        - Load environment variables
        - Set up OpenAI client
        - Initialize tracking variables
        """
        # TODO: Add your initialization code here
        pass
    
    def zero_shot_prompt(self, task: str, system_prompt: str = None) -> str:
        """
        Generate a zero-shot prompt for the given task.
        
        Args:
            task (str): The task to perform
            system_prompt (str): Optional system prompt
            
        Returns:
            str: The formatted zero-shot prompt
            
        TODO: Implement zero-shot prompting
        - Format the task as a direct question
        - Include system prompt if provided
        - Return the complete prompt
        """
        # TODO: Your code here
        pass
    
    def few_shot_prompt(self, task: str, examples: List[Dict[str, str]], system_prompt: str = None) -> str:
        """
        Generate a few-shot prompt with examples.
        
        Args:
            task (str): The task to perform
            examples (List[Dict]): List of example input/output pairs
            system_prompt (str): Optional system prompt
            
        Returns:
            str: The formatted few-shot prompt
            
        TODO: Implement few-shot prompting
        - Format examples as input → output pairs
        - Add the target task at the end
        - Include system prompt if provided
        """
        # TODO: Your code here
        pass
    
    def chain_of_thought_prompt(self, task: str, system_prompt: str = None) -> str:
        """
        Generate a chain-of-thought prompt for step-by-step reasoning.
        
        Args:
            task (str): The task to perform
            system_prompt (str): Optional system prompt
            
        Returns:
            str: The formatted chain-of-thought prompt
            
        TODO: Implement chain-of-thought prompting
        - Add reasoning instructions
        - Encourage step-by-step thinking
        - Include system prompt if provided
        """
        # TODO: Your code here
        pass
    
    def call_openai_api(self, prompt: str, system_prompt: str = None, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """
        Make a call to the OpenAI API.
        
        Args:
            prompt (str): The user prompt
            system_prompt (str): Optional system prompt
            model (str): Model to use
            
        Returns:
            Dict[str, Any]: API response with content, usage, etc.
            
        TODO: Implement API call
        - Handle rate limiting with exponential backoff
        - Track token usage and costs
        - Handle errors gracefully
        - Return response with metadata
        """
        # TODO: Your code here
        pass
    
    def evaluate_response(self, response: str, expected: str, task_type: str) -> Dict[str, Any]:
        """
        Evaluate the quality of the response.
        
        Args:
            response (str): AI response
            expected (str): Expected answer
            task_type (str): Type of task (classification, math, etc.)
            
        Returns:
            Dict[str, Any]: Evaluation metrics
            
        TODO: Implement response evaluation
        - Calculate accuracy based on task type
        - Estimate confidence level
        - Handle different evaluation strategies
        """
        evaluation = {
            "accuracy": 0.0,
            "confidence": 0.0,
            "notes": ""
        }
        
        # TODO: Your evaluation code here
        
        return evaluation
    
    def run_task(self, task: Dict[str, Any], technique: str) -> TaskResult:
        """
        Run a single task with the specified technique.
        
        Args:
            task (Dict): Task definition with input, expected output, etc.
            technique (str): Prompting technique to use
            
        Returns:
            TaskResult: Complete result of the task execution
            
        TODO: Implement task execution
        - Generate appropriate prompt based on technique
        - Call OpenAI API
        - Evaluate response
        - Calculate costs and timing
        - Return TaskResult object
        """
        # TODO: Your code here
        pass
    
    def run_comparison(self, tasks: List[Dict[str, Any]], techniques: List[str] = None) -> List[TaskResult]:
        """
        Run all tasks with all techniques and compare results.
        
        Args:
            tasks (List[Dict]): List of tasks to test
            techniques (List[str]): List of techniques to test
            
        Returns:
            List[TaskResult]: Results for all task/technique combinations
            
        TODO: Implement comparison testing
        - Run each task with each technique
        - Collect all results
        - Handle errors gracefully
        - Return comprehensive results
        """
        if techniques is None:
            techniques = ["zero_shot", "few_shot", "chain_of_thought"]
        
        results = []
        
        # TODO: Your comparison code here
        
        return results
    
    def generate_report(self, results: List[TaskResult], output_file: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report from the results.
        
        Args:
            results (List[TaskResult]): All task results
            output_file (str): Optional file to save report
            
        Returns:
            Dict[str, Any]: Complete report data
            
        TODO: Implement report generation
        - Calculate aggregate statistics
        - Compare technique performance
        - Generate recommendations
        - Save to file if specified
        """
        report = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(results),
                "total_cost": 0.0,
                "api_calls": len(results)
            },
            "techniques": {},
            "recommendations": []
        }
        
        # TODO: Your report generation code here
        
        if output_file:
            # TODO: Save report to file
            pass
        
        return report
    
    def interactive_mode(self):
        """
        Run the playground in interactive mode for experimentation.
        
        TODO: Implement interactive mode
        - Allow user to input custom tasks
        - Let user choose techniques
        - Display results in real-time
        - Provide helpful feedback
        """
        print("🤖 Interactive Prompt Engineering Playground")
        print("=" * 50)
        print("Enter 'quit' to exit, 'help' for commands")
        
        # TODO: Your interactive mode code here
        pass


def load_sample_tasks() -> List[Dict[str, Any]]:
    """Load sample tasks from JSON file."""
    try:
        with open("sample_tasks.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default tasks if file not found
        return [
            {
                "id": "sentiment_1",
                "task": "Classify the sentiment of this text: 'I love this product, it's amazing!'",
                "expected": "positive",
                "type": "classification",
                "examples": [
                    {"input": "I hate this movie", "output": "negative"},
                    {"input": "This is great", "output": "positive"}
                ]
            },
            {
                "id": "math_1", 
                "task": "If a train travels 120 km in 2 hours, what is its speed in km/h?",
                "expected": "60 km/h",
                "type": "math",
                "examples": [
                    {"input": "A car travels 100 km in 2 hours. Speed?", "output": "50 km/h"}
                ]
            },
            {
                "id": "creative_1",
                "task": "Write a short story about a robot learning to paint",
                "expected": "creative_narrative",
                "type": "creative",
                "examples": []
            }
        ]


def main():
    """Main function to test your prompt engineering playground."""
    parser = argparse.ArgumentParser(description="Prompt Engineering Playground")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--report", "-r", type=str, help="Generate report file")
    parser.add_argument("--task", "-t", type=str, help="Test specific task")
    args = parser.parse_args()
    
    print("🚀 Assignment 2: Prompt Engineering Playground")
    print("=" * 50)
    
    # TODO: Initialize PromptEngineer
    # engineer = PromptEngineer()
    
    if args.interactive:
        # TODO: Run interactive mode
        pass
    else:
        # Load sample tasks
        tasks = load_sample_tasks()
        
        # TODO: Run comparison tests
        # results = engineer.run_comparison(tasks)
        
        # TODO: Generate and display report
        # report = engineer.generate_report(results, args.report)
        
        print("✅ Testing completed!")
        print("💡 Try running with --interactive for experimentation")


# Test your implementation
if __name__ == "__main__":
    main()


# BONUS CHALLENGES (Optional):
# 1. Add support for different OpenAI models (GPT-4, etc.)
# 2. Implement temperature and other parameter testing
# 3. Create a web interface using Streamlit
# 4. Add support for custom evaluation metrics
# 5. Implement prompt template management
# 6. Add cost optimization features


# TESTING CHECKLIST:
# □ Script runs without errors
# □ API calls work correctly
# □ All prompting techniques implemented
# □ Response evaluation works
# □ Cost tracking functional
# □ Report generation complete
# □ Interactive mode works
# □ Error handling robust
# □ Code is well-commented and readable


# SUBMISSION:
# 1. Complete all TODO sections
# 2. Test with sample tasks
# 3. Generate and verify reports
# 4. Add your name and date at the top
# 5. Submit assignment.py and generated reports
# 6. Include screenshots of your playground in action 