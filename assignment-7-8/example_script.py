"""
Assignment 7: Example AI Agent with LangGraph
This demonstrates a working ReAct agent that can perform multi-step reasoning
using LangGraph for state management.
"""

import os
import json
import argparse
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph import StateGraph, END
from langgraph.graph import Graph

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

class ExampleReasoningAgent:
    """Example implementation of a ReAct reasoning agent"""
    
    def __init__(self):
        """Initialize the reasoning agent"""
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        
        self.max_iterations = 8
        
        # Create the workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow"""
        
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self.planning_node)
        workflow.add_node("executor", self.execution_node)
        workflow.add_node("reflector", self.reflection_node)
        workflow.add_node("finalizer", self.completion_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_edge("planner", "executor")
        workflow.add_conditional_edges(
            "executor",
            self.should_continue,
            {
                "reflect": "reflector",
                "continue": "executor",
                "finalize": "finalizer"
            }
        )
        workflow.add_edge("reflector", "executor")
        workflow.add_edge("finalizer", END)
        
        return workflow.compile()
    
    def planning_node(self, state: AgentState) -> AgentState:
        """Planning node: Break down the task into steps"""
        
        plan_prompt = f"""
        You are a reasoning agent. Break down this task into clear, logical steps:
        
        Task: {state['task']}
        
        Provide a numbered list of 3-5 specific steps that can be executed to solve this task.
        Each step should be concrete and actionable.
        
        Format your response as:
        1. [First step]
        2. [Second step]
        3. [Third step]
        etc.
        """
        
        messages = [
            SystemMessage(content="You are a helpful planning assistant."),
            HumanMessage(content=plan_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            plan_text = response.content
            
            # Parse the plan into steps
            plan = []
            for line in plan_text.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                    # Remove the number and add to plan
                    step = line[2:].strip()  # Remove "1. " part
                    if step:
                        plan.append(step)
            
            state['plan'] = plan
            print(f"📋 Generated plan with {len(plan)} steps")
            
        except Exception as e:
            print(f"Error in planning: {e}")
            state['plan'] = ["Solve the task step by step"]
        
        return state
    
    def execution_node(self, state: AgentState) -> AgentState:
        """Execution node: Execute current step using ReAct pattern"""
        
        current_step = state['current_step']
        
        if current_step >= len(state['plan']):
            state['complete'] = True
            return state
        
        step_description = state['plan'][current_step]
        
        # Generate thought
        thought = self._generate_thought(state, step_description)
        
        # Generate action
        action = self._generate_action(state, thought, step_description)
        
        # Execute action
        observation = self._execute_action(action, state)
        
        # Update state
        state['thoughts'].append(thought)
        state['actions'].append(action)
        state['observations'].append(observation)
        state['current_step'] += 1
        state['iteration_count'] += 1
        
        print(f"🧠 Step {current_step + 1}: {step_description}")
        print(f"   💭 Thought: {thought}")
        print(f"   🎯 Action: {action}")
        print(f"   👀 Observation: {observation}")
        
        return state
    
    def _generate_thought(self, state: AgentState, step_description: str) -> str:
        """Generate a thought for the current step"""
        
        context = ""
        if state['observations']:
            context = f"Previous observations: {'; '.join(state['observations'][-2:])}"
        
        thought_prompt = f"""
        Current task: {state['task']}
        Current step to complete: {step_description}
        {context}
        
        What should I think about to approach this step? Provide a brief thought about how to tackle this step.
        Keep it to 1-2 sentences.
        """
        
        messages = [
            SystemMessage(content="You are a reasoning agent generating thoughts."),
            HumanMessage(content=thought_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"I need to work on: {step_description}"
    
    def _generate_action(self, state: AgentState, thought: str, step_description: str) -> str:
        """Generate an action based on the thought"""
        
        action_prompt = f"""
        Thought: {thought}
        Step to complete: {step_description}
        
        Based on this thought, what specific action should I take to complete this step?
        Be concrete and specific. Keep it to 1-2 sentences.
        """
        
        messages = [
            SystemMessage(content="You are a reasoning agent determining actions."),
            HumanMessage(content=action_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Execute: {step_description}"
    
    def _execute_action(self, action: str, state: AgentState) -> str:
        """Execute the action and return observation"""
        
        # Determine the type of task for appropriate execution
        task_lower = state['task'].lower()
        
        if any(word in task_lower for word in ['math', 'calculate', 'solve', 'equation']):
            return self._execute_math_action(action, state)
        elif any(word in task_lower for word in ['logic', 'puzzle', 'who', 'which']):
            return self._execute_logic_action(action, state)
        else:
            return self._execute_general_action(action, state)
    
    def _execute_math_action(self, action: str, state: AgentState) -> str:
        """Execute mathematical reasoning action"""
        
        execution_prompt = f"""
        Task context: {state['task']}
        Action to execute: {action}
        Previous work: {'; '.join(state['observations'][-2:]) if state['observations'] else 'Starting fresh'}
        
        Execute this mathematical action step by step. Show your calculations and reasoning.
        If you need to perform calculations, do them explicitly.
        """
        
        messages = [
            SystemMessage(content="You are a mathematical reasoning assistant. Show your work clearly."),
            HumanMessage(content=execution_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Executed {action} - mathematical reasoning applied"
    
    def _execute_logic_action(self, action: str, state: AgentState) -> str:
        """Execute logical reasoning action"""
        
        execution_prompt = f"""
        Task context: {state['task']}
        Action to execute: {action}
        What we know so far: {'; '.join(state['observations'][-2:]) if state['observations'] else 'Starting analysis'}
        
        Execute this logical reasoning action. Consider the constraints and clues carefully.
        Show your logical deduction process.
        """
        
        messages = [
            SystemMessage(content="You are a logical reasoning assistant. Think step by step."),
            HumanMessage(content=execution_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Executed {action} - logical analysis applied"
    
    def _execute_general_action(self, action: str, state: AgentState) -> str:
        """Execute general reasoning action"""
        
        execution_prompt = f"""
        Task context: {state['task']}
        Action to execute: {action}
        Previous context: {'; '.join(state['observations'][-2:]) if state['observations'] else 'Beginning analysis'}
        
        Execute this action thoughtfully. Provide insights, analysis, or information relevant to the task.
        """
        
        messages = [
            SystemMessage(content="You are a general reasoning assistant."),
            HumanMessage(content=execution_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Executed {action} - analysis completed"
    
    def reflection_node(self, state: AgentState) -> AgentState:
        """Reflection node: Evaluate progress"""
        
        reflection_prompt = f"""
        Task: {state['task']}
        Plan: {state['plan']}
        
        Progress so far:
        {self._format_progress(state)}
        
        Reflect on the progress:
        1. Are we making good progress toward solving the task?
        2. Do we need to adjust our approach?
        3. What should we focus on in the next steps?
        
        Provide a brief reflection (2-3 sentences).
        """
        
        messages = [
            SystemMessage(content="You are a reflective reasoning assistant."),
            HumanMessage(content=reflection_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            reflection = response.content.strip()
            state['reflections'].append(reflection)
            
            print(f"🤔 Reflection: {reflection}")
            
        except Exception as e:
            reflection = "Continuing with current approach."
            state['reflections'].append(reflection)
        
        return state
    
    def completion_node(self, state: AgentState) -> AgentState:
        """Completion node: Generate final answer"""
        
        completion_prompt = f"""
        Original task: {state['task']}
        
        All work completed:
        {self._format_progress(state)}
        
        Based on all the reasoning and analysis above, provide a comprehensive final answer to the original task.
        Be clear, concise, and complete in your response.
        """
        
        messages = [
            SystemMessage(content="You are completing a reasoning task. Provide a clear final answer."),
            HumanMessage(content=completion_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            final_answer = response.content.strip()
            state['final_answer'] = final_answer
            state['complete'] = True
            
            print(f"✅ Final Answer: {final_answer}")
            
        except Exception as e:
            state['final_answer'] = "Unable to generate final answer due to error."
            state['complete'] = True
        
        return state
    
    def should_continue(self, state: AgentState) -> str:
        """Determine next node in workflow"""
        
        if state['complete']:
            return "finalize"
        
        if state['iteration_count'] >= self.max_iterations:
            return "finalize"
        
        if state['current_step'] >= len(state['plan']):
            return "finalize"
        
        # Reflect every 3 steps
        if state['current_step'] > 0 and state['current_step'] % 3 == 0:
            return "reflect"
        
        return "continue"
    
    def _format_progress(self, state: AgentState) -> str:
        """Format progress for display"""
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
    
    def solve_problem(self, task: str) -> Dict[str, Any]:
        """Solve a problem using the agent"""
        
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
        
        print(f"🎯 Starting task: {task}")
        print("="*60)
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            return {
                'task': task,
                'final_answer': final_state.get('final_answer', 'No answer generated'),
                'reasoning_trace': self._format_progress(final_state),
                'success': final_state.get('complete', False),
                'steps_completed': final_state.get('current_step', 0),
                'total_iterations': final_state.get('iteration_count', 0)
            }
            
        except Exception as e:
            print(f"❌ Error running workflow: {e}")
            return {
                'task': task,
                'final_answer': f'Error: {e}',
                'reasoning_trace': '',
                'success': False
            }

def load_sample_tasks() -> Dict[str, List[str]]:
    """Load sample tasks for testing"""
    
    return {
        'math': [
            "If a train travels 120 km in 2 hours, then 180 km in 3 hours, what was the average speed for the entire journey?",
            "A rectangle has a length that is 3 times its width. If the perimeter is 48 cm, what are the dimensions?",
            "If you invest $1000 at 5% annual interest compounded annually, how much will you have after 3 years?"
        ],
        'logic': [
            "Three friends - Alice, Bob, and Charlie - have different pets (cat, dog, bird) and live in different colored houses (red, blue, green). Alice doesn't have a cat. The person with the bird lives in the blue house. Bob lives in the red house. Who has which pet and lives where?",
            "A farmer has chickens and rabbits. In total, there are 20 heads and 56 legs. How many chickens and rabbits are there?",
            "You have 3 boxes. One contains only apples, one contains only oranges, and one contains both. All boxes are labeled incorrectly. You can pick one fruit from one box. How can you correctly label all boxes?"
        ],
        'research': [
            "Compare the advantages and disadvantages of solar energy vs wind energy for residential use.",
            "Analyze the potential impact of artificial intelligence on the job market in the next decade.",
            "What are the key factors to consider when choosing between renting and buying a home?"
        ]
    }

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Example AI Reasoning Agent')
    parser.add_argument('--task', choices=['math', 'logic', 'research'], 
                       help='Type of task to solve')
    parser.add_argument('--custom', type=str, help='Custom problem to solve')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--verbose', action='store_true', 
                       help='Show detailed reasoning trace')
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return
    
    # Initialize agent
    print("🤖 Initializing AI Reasoning Agent...")
    try:
        agent = ExampleReasoningAgent()
        print("✅ Agent ready!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Load sample tasks
    sample_tasks = load_sample_tasks()
    
    if args.interactive:
        print("\n🔄 Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            task = input("💬 Enter a problem to solve: ").strip()
            if task.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not task:
                continue
            
            result = agent.solve_problem(task)
            
            print(f"\n🎯 Final Answer: {result['final_answer']}")
            print(f"✅ Success: {result['success']}")
            
            if args.verbose and result['reasoning_trace']:
                print(f"\n📝 Reasoning Trace:\n{result['reasoning_trace']}")
    
    elif args.custom:
        print(f"🧠 Solving custom problem...")
        result = agent.solve_problem(args.custom)
        
        print(f"\n🎯 Answer: {result['final_answer']}")
        
        if args.verbose:
            print(f"\n📝 Reasoning Trace:\n{result['reasoning_trace']}")
    
    elif args.task:
        task_type = args.task
        if task_type not in sample_tasks:
            print(f"Unknown task type: {task_type}")
            return
        
        print(f"\n🧠 Testing {task_type} problems:")
        print("="*60)
        
        for i, problem in enumerate(sample_tasks[task_type], 1):
            print(f"\n📝 Problem {i}: {problem}")
            print("-" * 50)
            
            result = agent.solve_problem(problem)
            
            print(f"📊 Steps completed: {result.get('steps_completed', 0)}")
            print(f"🔄 Total iterations: {result.get('total_iterations', 0)}")
            
            if args.verbose and result['reasoning_trace']:
                print(f"\n📋 Full reasoning trace:\n{result['reasoning_trace']}")
    
    else:
        print("🤖 AI Reasoning Agent - Example Implementation")
        print("\nUsage examples:")
        print("  python example_agent.py --task math")
        print("  python example_agent.py --custom 'Your problem here'")
        print("  python example_agent.py --interactive")
        print("  python example_agent.py --task logic --verbose")

if __name__ == "__main__":
    main() 