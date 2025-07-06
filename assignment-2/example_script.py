#!/usr/bin/env python3
"""
Assignment 2: Prompt Engineering Playground Example Script
Author: GenAI Bootcamp
Description: Complete implementation of prompt engineering playground with OpenAI API
"""

import os
import json
import time
import argparse
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# OpenAI API imports
import openai
from dotenv import load_dotenv


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
    """A comprehensive prompt engineering playground for testing different techniques."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the PromptEngineer with OpenAI API access.
        
        Args:
            api_key (str): OpenAI API key (optional, can use environment variable)
        """
        # Load environment variables
        load_dotenv()
        
        # Set up OpenAI client
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize tracking variables
        self.total_cost = 0.0
        self.total_calls = 0
        self.rate_limit_delay = 1.0  # seconds between calls
        
        # Cost per 1K tokens (gpt-4o-mini)
        self.cost_per_1k_input = 0.0015
        self.cost_per_1k_output = 0.002
        
        print(f"✅ PromptEngineer initialized with API key: {openai.api_key[:8]}...")
    
    def zero_shot_prompt(self, task: str, system_prompt: str = None) -> str:
        """
        Generate a zero-shot prompt for the given task.
        
        Args:
            task (str): The task to perform
            system_prompt (str): Optional system prompt
            
        Returns:
            str: The formatted zero-shot prompt
        """
        if system_prompt:
            return f"{system_prompt}\n\n{task}"
        return task
    
    def few_shot_prompt(self, task: str, examples: List[Dict[str, str]], system_prompt: str = None) -> str:
        """
        Generate a few-shot prompt with examples.
        
        Args:
            task (str): The task to perform
            examples (List[Dict]): List of example input/output pairs
            system_prompt (str): Optional system prompt
            
        Returns:
            str: The formatted few-shot prompt
        """
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(system_prompt)
        
        # Add examples
        for example in examples:
            prompt_parts.append(f"Input: {example['input']}")
            prompt_parts.append(f"Output: {example['output']}")
            prompt_parts.append("")  # Empty line for separation
        
        # Add the target task
        prompt_parts.append(f"Input: {task}")
        prompt_parts.append("Output:")
        
        return "\n".join(prompt_parts)
    
    def chain_of_thought_prompt(self, task: str, system_prompt: str = None) -> str:
        """
        Generate a chain-of-thought prompt for step-by-step reasoning.
        
        Args:
            task (str): The task to perform
            system_prompt (str): Optional system prompt
            
        Returns:
            str: The formatted chain-of-thought prompt
        """
        cot_instruction = "Let's solve this step by step. Think through the problem carefully and show your reasoning."
        
        if system_prompt:
            base_prompt = f"{system_prompt}\n\n{cot_instruction}\n\n{task}"
        else:
            base_prompt = f"{cot_instruction}\n\n{task}"
        
        return base_prompt
    
    def call_openai_api(self, prompt: str, system_prompt: str = None, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Make a call to the OpenAI API with rate limiting and error handling.
        
        Args:
            prompt (str): The user prompt
            system_prompt (str): Optional system prompt
            model (str): Model to use
            
        Returns:
            Dict[str, Any]: API response with content, usage, etc.
        """
        # Rate limiting
        time.sleep(self.rate_limit_delay)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                
                # Calculate cost
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = (input_tokens / 1000 * self.cost_per_1k_input + 
                       output_tokens / 1000 * self.cost_per_1k_output)
                
                self.total_cost += cost
                self.total_calls += 1
                
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "cost": cost,
                    "model": model
                }
                
            except openai.error.RateLimitError:
                if attempt < max_retries - 1:
                    print(f"⚠️  Rate limit hit, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception("Rate limit exceeded after multiple retries")
                    
            except openai.error.APIError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  API error: {e}, retrying...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"API error after multiple retries: {e}")
                    
            except Exception as e:
                raise Exception(f"Unexpected error: {e}")
    
    def evaluate_response(self, response: str, expected: str, task_type: str) -> Dict[str, Any]:
        """
        Evaluate the quality of the response.
        
        Args:
            response (str): AI response
            expected (str): Expected answer
            task_type (str): Type of task (classification, math, etc.)
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        evaluation = {
            "accuracy": 0.0,
            "confidence": 0.0,
            "notes": ""
        }
        
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()
        
        if task_type == "classification":
            # Simple exact match for classification
            if expected_lower in response_lower or response_lower in expected_lower:
                evaluation["accuracy"] = 1.0
                evaluation["confidence"] = 0.9
            else:
                evaluation["accuracy"] = 0.0
                evaluation["confidence"] = 0.3
                
        elif task_type == "math":
            # Extract numbers from response
            numbers_in_response = re.findall(r'\d+(?:\.\d+)?', response)
            numbers_in_expected = re.findall(r'\d+(?:\.\d+)?', expected)
            
            if numbers_in_response and numbers_in_expected:
                response_num = float(numbers_in_response[0])
                expected_num = float(numbers_in_expected[0])
                
                if abs(response_num - expected_num) < 0.1:  # Allow small tolerance
                    evaluation["accuracy"] = 1.0
                    evaluation["confidence"] = 0.8
                else:
                    evaluation["accuracy"] = 0.0
                    evaluation["confidence"] = 0.4
            else:
                evaluation["accuracy"] = 0.0
                evaluation["confidence"] = 0.2
                
        elif task_type == "creative":
            # For creative tasks, check if response is substantial
            if len(response) > 50 and not response.startswith("I don't"):
                evaluation["accuracy"] = 0.8  # Creative tasks are subjective
                evaluation["confidence"] = 0.7
            else:
                evaluation["accuracy"] = 0.0
                evaluation["confidence"] = 0.3
        else:
            # Generic evaluation
            if expected_lower in response_lower:
                evaluation["accuracy"] = 0.8
                evaluation["confidence"] = 0.6
            else:
                evaluation["accuracy"] = 0.0
                evaluation["confidence"] = 0.3
        
        return evaluation
    
    def run_task(self, task: Dict[str, Any], technique: str) -> TaskResult:
        """
        Run a single task with the specified technique.
        
        Args:
            task (Dict): Task definition with input, expected output, etc.
            technique (str): Prompting technique to use
            
        Returns:
            TaskResult: Complete result of the task execution
        """
        task_id = task["id"]
        task_text = task["task"]
        expected = task["expected"]
        task_type = task.get("type", "general")
        examples = task.get("examples", [])
        
        print(f"🔄 Running {technique} on task: {task_id}")
        
        # Generate appropriate prompt
        if technique == "zero_shot":
            prompt = self.zero_shot_prompt(task_text)
        elif technique == "few_shot":
            prompt = self.few_shot_prompt(task_text, examples)
        elif technique == "chain_of_thought":
            prompt = self.chain_of_thought_prompt(task_text)
        else:
            raise ValueError(f"Unknown technique: {technique}")
        
        # Call API
        start_time = time.time()
        api_response = self.call_openai_api(prompt)
        response_time = time.time() - start_time
        
        # Evaluate response
        evaluation = self.evaluate_response(api_response["content"], expected, task_type)
        
        # Create result
        result = TaskResult(
            task_id=task_id,
            technique=technique,
            prompt=prompt,
            response=api_response["content"],
            expected=expected,
            cost=api_response["cost"],
            response_time=response_time,
            accuracy=evaluation["accuracy"],
            confidence=evaluation["confidence"]
        )
        
        # Display result
        print(f"✅ {technique.title()} - {task_id}")
        print(f"   Response: {api_response['content'][:100]}...")
        print(f"   Accuracy: {evaluation['accuracy']:.1%}")
        print(f"   Cost: ${api_response['cost']:.4f}")
        print(f"   Time: {response_time:.2f}s")
        print()
        
        return result
    
    def run_comparison(self, tasks: List[Dict[str, Any]], techniques: List[str] = None) -> List[TaskResult]:
        """
        Run all tasks with all techniques and compare results.
        
        Args:
            tasks (List[Dict]): List of tasks to test
            techniques (List[str]): List of techniques to test
            
        Returns:
            List[TaskResult]: Results for all task/technique combinations
        """
        if techniques is None:
            techniques = ["zero_shot", "few_shot", "chain_of_thought"]
        
        results = []
        
        print(f"🚀 Starting comparison with {len(tasks)} tasks and {len(techniques)} techniques")
        print("=" * 60)
        
        for technique in techniques:
            print(f"\n📊 Testing {technique.replace('_', ' ').title()} Technique")
            print("-" * 40)
            
            for task in tasks:
                try:
                    result = self.run_task(task, technique)
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error running {technique} on {task['id']}: {e}")
                    continue
        
        return results
    
    def generate_report(self, results: List[TaskResult], output_file: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report from the results.
        
        Args:
            results (List[TaskResult]): All task results
            output_file (str): Optional file to save report
            
        Returns:
            Dict[str, Any]: Complete report data
        """
        # Calculate aggregate statistics
        total_cost = sum(r.cost for r in results)
        total_time = sum(r.response_time for r in results)
        
        # Group by technique
        technique_stats = {}
        for technique in set(r.technique for r in results):
            technique_results = [r for r in results if r.technique == technique]
            technique_stats[technique] = {
                "count": len(technique_results),
                "avg_accuracy": sum(r.accuracy for r in technique_results) / len(technique_results),
                "avg_confidence": sum(r.confidence for r in technique_results) / len(technique_results),
                "avg_cost": sum(r.cost for r in technique_results) / len(technique_results),
                "avg_time": sum(r.response_time for r in technique_results) / len(technique_results),
                "total_cost": sum(r.cost for r in technique_results)
            }
        
        # Generate recommendations
        recommendations = []
        best_accuracy = max(stats["avg_accuracy"] for stats in technique_stats.values())
        best_technique = [k for k, v in technique_stats.items() if v["avg_accuracy"] == best_accuracy][0]
        
        recommendations.append(f"Best performing technique: {best_technique} ({best_accuracy:.1%} accuracy)")
        
        if technique_stats.get("few_shot", {}).get("avg_accuracy", 0) > 0.8:
            recommendations.append("Few-shot prompting works well for structured tasks")
        
        if technique_stats.get("chain_of_thought", {}).get("avg_accuracy", 0) > 0.7:
            recommendations.append("Chain-of-thought is effective for reasoning problems")
        
        cost_per_accuracy = {k: v["avg_cost"] / v["avg_accuracy"] if v["avg_accuracy"] > 0 else float('inf') 
                           for k, v in technique_stats.items()}
        most_efficient = min(cost_per_accuracy.items(), key=lambda x: x[1])[0]
        recommendations.append(f"Most cost-effective: {most_efficient}")
        
        report = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(results),
                "total_cost": total_cost,
                "api_calls": len(results),
                "avg_response_time": total_time / len(results) if results else 0
            },
            "techniques": technique_stats,
            "recommendations": recommendations,
            "detailed_results": [asdict(r) for r in results]
        }
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"✅ Report saved to: {output_file}")
            except Exception as e:
                print(f"❌ Error saving report: {e}")
        
        return report
    
    def interactive_mode(self):
        """
        Run the playground in interactive mode for experimentation.
        """
        print("🤖 Interactive Prompt Engineering Playground")
        print("=" * 50)
        print("Commands: 'quit' to exit, 'help' for commands, 'stats' for session stats")
        
        while True:
            try:
                user_input = input("\n🎯 Enter your task (or command): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    print("\n📖 Available commands:")
                    print("  - Enter any task to test it")
                    print("  - 'stats' - Show session statistics")
                    print("  - 'quit' - Exit the playground")
                    print("  - 'help' - Show this help")
                    continue
                elif user_input.lower() == 'stats':
                    print(f"\n📊 Session Statistics:")
                    print(f"  Total API calls: {self.total_calls}")
                    print(f"  Total cost: ${self.total_cost:.4f}")
                    continue
                elif not user_input:
                    continue
                
                # Test the user's task
                print(f"\n🔄 Testing your task with different techniques...")
                
                # Create a simple task
                task = {
                    "id": "user_task",
                    "task": user_input,
                    "expected": "user_response",
                    "type": "general",
                    "examples": []
                }
                
                # Test with different techniques
                techniques = ["zero_shot", "few_shot", "chain_of_thought"]
                for technique in techniques:
                    try:
                        result = self.run_task(task, technique)
                        print(f"💡 {technique.title()} response: {result.response[:200]}...")
                    except Exception as e:
                        print(f"❌ Error with {technique}: {e}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


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
            },
            {
                "id": "code_1",
                "task": "Write a Python function to calculate fibonacci numbers",
                "expected": "python_code",
                "type": "code",
                "examples": [
                    {"input": "Write a function to add two numbers", "output": "def add(a, b): return a + b"}
                ]
            }
        ]


def main():
    """Main function to demonstrate the prompt engineering playground."""
    parser = argparse.ArgumentParser(description="Prompt Engineering Playground")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--report", "-r", type=str, help="Generate report file")
    parser.add_argument("--task", "-t", type=str, help="Test specific task")
    args = parser.parse_args()
    
    print("🚀 GenAI Bootcamp - Assignment 2: Prompt Engineering Playground")
    print("=" * 60)
    
    try:
        # Initialize PromptEngineer
        engineer = PromptEngineer()
        
        if args.interactive:
            engineer.interactive_mode()
        else:
            # Load sample tasks
            tasks = load_sample_tasks()
            
            if args.task:
                # Test specific task
                custom_task = {
                    "id": "custom_task",
                    "task": args.task,
                    "expected": "custom_response",
                    "type": "general",
                    "examples": []
                }
                tasks = [custom_task]
            
            # Run comparison tests
            results = engineer.run_comparison(tasks)
            
            # Generate and display report
            report = engineer.generate_report(results, args.report)
            
            # Display summary
            print("\n📊 Summary Report:")
            print("-" * 30)
            print(f"Total API calls: {report['session_info']['api_calls']}")
            print(f"Total cost: ${report['session_info']['total_cost']:.4f}")
            print(f"Average response time: {report['session_info']['avg_response_time']:.2f}s")
            
            print("\n🎯 Technique Performance:")
            for technique, stats in report['techniques'].items():
                print(f"  {technique.replace('_', ' ').title()}: {stats['avg_accuracy']:.1%} accuracy, ${stats['avg_cost']:.4f} avg cost")
            
            print("\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
            
            print(f"\n✅ Testing completed!")
            print("💡 Try running with --interactive for experimentation")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have set up your OpenAI API key in the .env file")


if __name__ == "__main__":
    main() 