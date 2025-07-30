"""
Assignment 7: AI Agents & LangGraph Basics
Student Name: [Your Name Here]
Date: [Date]

Instructions:
1. Implement a ReAct agent using LangGraph
2. Create state management for multi-step reasoning
3. Add reflection and self-correction capabilities
4. Test with different types of reasoning tasks
5. Document the agent's reasoning process
"""

import os
import json
import argparse
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# TODO: Import LangChain and LangGraph components
# Hint: You'll need ChatOpenAI, StateGraph, and other components

load_dotenv()

# Agent State Definition
class AgentState(TypedDict):
    """State structure for the reasoning agent"""
    task: str
    plan: List[str]
    current_step: int
    thoughts: List[str]
    actions: List[str]
    observations: List[str]
    reflections: List[str]
    final_answer: Optional[str]
    complete: bool
    iteration_count: int

@dataclass
class ReasoningStep:
    """Represents a single reasoning step"""
    thought: str
    action: str
    observation: str
    step_number: int

class ReasoningAgent:
    """AI Agent that can perform multi-step reasoning using the ReAct pattern"""
    
    def __init__(self):
        """Initialize the reasoning agent"""
        
        # TODO: Initialize OpenAI LLM
        # Hint: Use ChatOpenAI from langchain_openai
        
        # TODO: Initialize LangGraph workflow
        # Hint: Use StateGraph with AgentState
        
        self.max_iterations = 10
        self.workflow = None
        
        # Set up the agent workflow
        self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for the agent"""
        
        # TODO: Create StateGraph workflow
        # Add nodes for: planning, execution, reflection, completion
        # Add appropriate edges and conditional logic
        
        # Example structure:
        # workflow = StateGraph(AgentState)
        # workflow.add_node("planner", self.planning_node)
        # workflow.add_node("executor", self.execution_node)
        # workflow.add_node("reflector", self.reflection_node)
        # workflow.add_node("finalizer", self.completion_node)
        
        pass
    
    def planning_node(self, state: AgentState) -> AgentState:
        """
        Planning node: Break down the task into steps
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with plan
        """
        # TODO: Generate a plan for the given task
        # Use the LLM to break down the task into logical steps
        # Update the state with the generated plan
        
        # Example prompt structure:
        # "Break down this task into clear, logical steps: {task}"
        
        plan_prompt = f"""
        Task: {state['task']}
        
        Break this task down into clear, logical steps that can be executed one by one.
        Provide a numbered list of steps.
        """
        
        # TODO: Use LLM to generate plan
        # TODO: Parse the response and extract steps
        # TODO: Update state with plan
        
        return state
    
    def execution_node(self, state: AgentState) -> AgentState:
        """
        Execution node: Execute the current step using ReAct pattern
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with execution results
        """
        current_step = state['current_step']
        
        if current_step >= len(state['plan']):
            state['complete'] = True
            return state
        
        # TODO: Implement ReAct pattern for current step
        # 1. Generate thought about what to do
        # 2. Determine action to take
        # 3. Execute action and observe result
        # 4. Update state with results
        
        # Get current step to execute
        step_description = state['plan'][current_step]
        
        # TODO: Generate thought
        thought_prompt = f"""
        Current task: {state['task']}
        Current step: {step_description}
        Previous observations: {state['observations'][-3:] if state['observations'] else 'None'}
        
        What should I think about to complete this step?
        """
        
        # TODO: Generate action
        action_prompt = f"""
        Thought: {thought}
        Step to complete: {step_description}
        
        What specific action should I take to complete this step?
        """
        
        # TODO: Execute action and get observation
        observation = self._execute_action(action, state)
        
        # TODO: Update state
        state['thoughts'].append(thought)
        state['actions'].append(action)
        state['observations'].append(observation)
        state['current_step'] += 1
        
        return state
    
    def _execute_action(self, action: str, state: AgentState) -> str:
        """
        Execute a specific action and return the observation
        
        Args:
            action: Action to execute
            state: Current agent state
            
        Returns:
            Observation from executing the action
        """
        # TODO: Implement action execution
        # This could involve:
        # - Mathematical calculations
        # - Logical reasoning
        # - Information analysis
        # - Text processing
        
        # For now, use LLM to simulate action execution
        execution_prompt = f"""
        Task context: {state['task']}
        Action to execute: {action}
        Previous context: {state['observations'][-2:] if state['observations'] else 'None'}
        
        Execute this action and provide the result/observation.
        """
        
        # TODO: Use LLM to execute action
        observation = "Action executed - provide actual implementation"
        
        return observation
    
    def reflection_node(self, state: AgentState) -> AgentState:
        """
        Reflection node: Evaluate progress and potentially adjust approach
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with reflection
        """
        # TODO: Implement reflection logic
        # Evaluate if the agent is making progress
        # Determine if plan needs adjustment
        # Decide whether to continue or change approach
        
        reflection_prompt = f"""
        Task: {state['task']}
        Plan: {state['plan']}
        Progress so far:
        {self._format_progress(state)}
        
        Reflect on the progress:
        1. Are we making good progress toward the goal?
        2. Do we need to adjust our approach?
        3. What should we focus on next?
        """
        
        # TODO: Generate reflection using LLM
        reflection = "Generated reflection - implement actual logic"
        
        state['reflections'].append(reflection)
        
        # TODO: Determine if plan needs adjustment
        # TODO: Implement logic to modify plan if needed
        
        return state
    
    def completion_node(self, state: AgentState) -> AgentState:
        """
        Completion node: Generate final answer
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with final answer
        """
        # TODO: Generate final answer based on all observations
        
        completion_prompt = f"""
        Task: {state['task']}
        All observations and progress:
        {self._format_progress(state)}
        
        Based on all the work done, provide a comprehensive final answer to the original task.
        """
        
        # TODO: Generate final answer using LLM
        final_answer = "Generated final answer - implement actual logic"
        
        state['final_answer'] = final_answer
        state['complete'] = True
        
        return state
    
    def _format_progress(self, state: AgentState) -> str:
        """Format the agent's progress for display or prompts"""
        progress = []
        
        for i, (thought, action, observation) in enumerate(
            zip(state['thoughts'], state['actions'], state['observations'])
        ):
            progress.append(f"Step {i+1}:")
            progress.append(f"  Thought: {thought}")
            progress.append(f"  Action: {action}")
            progress.append(f"  Observation: {observation}")
            progress.append("")
        
        return "\n".join(progress)
    
    def should_continue(self, state: AgentState) -> str:
        """
        Determine next node in the workflow
        
        Args:
            state: Current agent state
            
        Returns:
            Name of next node to execute
        """
        # TODO: Implement conditional logic for workflow
        
        if state['complete']:
            return "finalizer"
        
        if state['iteration_count'] >= self.max_iterations:
            return "finalizer"
        
        if state['current_step'] >= len(state['plan']):
            return "finalizer"
        
        # Check if reflection is needed (every few steps)
        if state['current_step'] > 0 and state['current_step'] % 3 == 0:
            return "reflector"
        
        return "executor"
    
    def solve_problem(self, task: str) -> Dict[str, Any]:
        """
        Solve a problem using the agent
        
        Args:
            task: Problem description
            
        Returns:
            Dictionary with solution and reasoning trace
        """
        # Initialize state
        initial_state = AgentState(
            task=task,
            plan=[],
            current_step=0,
            thoughts=[],
            actions=[],
            observations=[],
            reflections=[],
            final_answer=None,
            complete=False,
            iteration_count=0
        )
        
        # TODO: Run the workflow
        # Use self.workflow.invoke(initial_state) or similar
        
        # For now, return a placeholder
        result = {
            'task': task,
            'final_answer': 'Implement workflow execution',
            'reasoning_trace': self._format_progress(initial_state),
            'success': False
        }
        
        return result

def load_sample_tasks() -> Dict[str, List[str]]:
    """Load sample tasks for testing"""
    
    tasks = {
        'math': [
            "Solve step by step: If a train travels 120 km in 2 hours, then 180 km in 3 hours, what was the average speed for the entire journey?",
            "A rectangle has a length that is 3 times its width. If the perimeter is 48 cm, what are the dimensions?",
            "If you invest $1000 at 5% annual interest compounded annually, how much will you have after 3 years?"
        ],
        'logic': [
            "Three friends have different pets (cat, dog, bird) and live in different colored houses (red, blue, green). Alice doesn't have a cat. The person with the bird lives in the blue house. Bob lives in the red house. Who has which pet and lives where?",
            "A farmer has chickens and rabbits. In total, there are 20 heads and 56 legs. How many chickens and rabbits are there?",
            "If all roses are flowers, and some flowers are red, can we conclude that some roses are red?"
        ],
        'research': [
            "Compare the advantages and disadvantages of solar energy vs wind energy for residential use.",
            "Analyze the potential impact of artificial intelligence on the job market in the next decade.",
            "Evaluate the pros and cons of remote work vs office work for software development teams."
        ]
    }
    
    return tasks

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='AI Reasoning Agent')
    parser.add_argument('--task', choices=['math', 'logic', 'research'], 
                       help='Type of task to solve')
    parser.add_argument('--custom', type=str, help='Custom problem to solve')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--verbose', action='store_true', 
                       help='Show detailed reasoning trace')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ReasoningAgent()
    
    # Load sample tasks
    sample_tasks = load_sample_tasks()
    
    if args.interactive:
        print("🤖 AI Reasoning Agent - Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            task = input("Enter a problem to solve: ").strip()
            if task.lower() in ['quit', 'exit', 'q']:
                break
            
            if not task:
                continue
            
            print(f"\n🧠 Solving: {task}")
            print("=" * 50)
            
            result = agent.solve_problem(task)
            
            print(f"🎯 Final Answer: {result['final_answer']}")
            
            if args.verbose:
                print(f"\n📝 Reasoning Trace:\n{result['reasoning_trace']}")
    
    elif args.custom:
        print(f"🧠 Solving custom problem: {args.custom}")
        result = agent.solve_problem(args.custom)
        print(f"🎯 Answer: {result['final_answer']}")
        
        if args.verbose:
            print(f"\n📝 Reasoning Trace:\n{result['reasoning_trace']}")
    
    elif args.task:
        task_type = args.task
        if task_type not in sample_tasks:
            print(f"Unknown task type: {task_type}")
            return
        
        print(f"🧠 Testing {task_type} problems:")
        print("=" * 50)
        
        for i, problem in enumerate(sample_tasks[task_type], 1):
            print(f"\nProblem {i}: {problem}")
            print("-" * 40)
            
            result = agent.solve_problem(problem)
            print(f"Answer: {result['final_answer']}")
            
            if args.verbose:
                print(f"Reasoning Trace:\n{result['reasoning_trace']}")
    
    else:
        print("🤖 AI Reasoning Agent")
        print("Usage examples:")
        print("  python assignment.py --task math")
        print("  python assignment.py --custom 'Your problem here'")
        print("  python assignment.py --interactive")
        print("  python assignment.py --task logic --verbose")

if __name__ == "__main__":
    main() 