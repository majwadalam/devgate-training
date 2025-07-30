# Assignment 7: AI Agents & LangGraph Basics

## Overview
Build an agent that can perform multi-step reasoning tasks using LangGraph. This assignment focuses on understanding agent architecture, the ReAct pattern, and creating state-based reasoning workflows.

## Learning Objectives
- Understand AI agent concepts and architecture patterns
- Learn the ReAct (Reasoning + Acting) pattern
- Work with LangGraph for creating state-based workflows
- Implement multi-step reasoning capabilities
- Build agents that can plan, execute, and reflect on tasks

## Setup Instructions

### 1. Create Virtual Environment
```bash
# Navigate to assignment-7 directory
cd assignment-7

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# OR
venv\Scripts\activate  # Windows CMD/PowerShell
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Assignment Requirements

### Task Description
Create an AI agent that can:

1. **Plan Tasks** - Break down complex problems into steps
2. **Execute Actions** - Perform individual reasoning steps
3. **Reflect on Results** - Evaluate progress and adjust approach
4. **Handle Multi-step Problems** - Solve tasks requiring multiple reasoning steps
5. **Maintain State** - Track progress through complex workflows

### Expected Features

#### Core Features (Required):
- [ ] Implement a basic ReAct agent structure
- [ ] Create state management using LangGraph
- [ ] Build multi-step reasoning capabilities
- [ ] Add reflection and self-correction mechanisms
- [ ] Handle different types of reasoning tasks

#### Bonus Features (Optional):
- [ ] Add memory persistence across sessions
- [ ] Implement planning with backtracking
- [ ] Create multiple specialized agent roles
- [ ] Add human-in-the-loop interaction
- [ ] Build agent collaboration capabilities

### Agent Architecture

#### 1. ReAct Pattern:
```
Thought: What do I need to do?
Action: [action to take]
Observation: [result of action]
Thought: What does this tell me?
Action: [next action]
...
```

#### 2. State Management:
- **Planning State**: Breaking down the task
- **Execution State**: Performing actions
- **Reflection State**: Evaluating results
- **Completion State**: Final answer or conclusion

#### 3. Multi-step Reasoning:
- Mathematical problem solving
- Research and analysis tasks
- Creative problem solving
- Logical deduction

## Files Structure

```
assignment-7/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── assignment.py          # Your solution template
├── example_agent.py       # Working example
├── tasks/                 # Sample reasoning tasks
│   ├── math_problems.json
│   ├── logic_puzzles.json
│   └── research_tasks.json
├── tools/                 # Agent tools and utilities
│   ├── calculator.py
│   ├── text_analyzer.py
│   └── web_search.py
└── venv/                  # Virtual environment
```

## Getting Started

1. **Run the example agent** to understand the pattern:
   ```bash
   python example_agent.py
   ```

2. **Try different reasoning tasks**:
   ```bash
   python assignment.py --task math
   python assignment.py --task logic
   python assignment.py --task research
   ```

3. **Implement your agent** in `assignment.py`

4. **Test with custom problems**:
   ```bash
   python assignment.py --custom "Solve the following problem step by step..."
   ```

## Key Concepts

### 1. Agent State
```python
from typing import TypedDict, List

class AgentState(TypedDict):
    task: str
    plan: List[str]
    current_step: int
    observations: List[str]
    reflections: List[str]
    final_answer: str
```

### 2. LangGraph Workflow
```python
from langgraph import StateGraph

# Create workflow
workflow = StateGraph(AgentState)
workflow.add_node("planner", planning_node)
workflow.add_node("executor", execution_node)
workflow.add_node("reflector", reflection_node)
```

### 3. ReAct Implementation
```python
def react_step(state: AgentState) -> AgentState:
    thought = generate_thought(state)
    action = generate_action(thought)
    observation = execute_action(action)
    return update_state(state, thought, action, observation)
```

## Sample Tasks

### 1. Mathematical Reasoning:
```
"Solve this step by step: If a train travels 120 km in 2 hours, 
and then 180 km in 3 hours, what was the average speed for 
the entire journey?"
```

### 2. Logic Puzzles:
```
"Three friends - Alice, Bob, and Charlie - each have a different 
pet (cat, dog, bird) and live in different colored houses 
(red, blue, green). Given these clues, determine who has 
which pet and lives in which house..."
```

### 3. Research Tasks:
```
"Analyze the pros and cons of renewable energy sources. 
Consider economic, environmental, and practical factors."
```

## Implementation Guide

### Step 1: Basic Agent Structure
```python
class ReasoningAgent:
    def __init__(self):
        self.llm = ChatOpenAI()
        self.state = AgentState()
        
    def plan(self, task: str) -> List[str]:
        # Break down task into steps
        pass
        
    def execute_step(self, step: str) -> str:
        # Execute a single reasoning step
        pass
        
    def reflect(self, observations: List[str]) -> str:
        # Reflect on progress and next steps
        pass
```

### Step 2: LangGraph Integration
```python
def create_agent_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("start", start_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("end", end_node)
    
    # Add edges
    workflow.add_edge("start", "plan")
    workflow.add_conditional_edges("execute", should_continue)
    
    return workflow.compile()
```

### Step 3: Multi-step Reasoning
```python
def solve_multi_step_problem(agent, problem):
    state = agent.initialize_state(problem)
    
    while not state.get("complete", False):
        state = agent.reasoning_step(state)
        
        if agent.should_reflect(state):
            state = agent.reflect_on_progress(state)
            
    return state["final_answer"]
```

## Evaluation Criteria

### 1. Reasoning Quality:
- Logical consistency of steps
- Accuracy of final answers
- Handling of edge cases

### 2. Agent Behavior:
- Appropriate planning strategies
- Effective self-correction
- Clear reasoning traces

### 3. Technical Implementation:
- Proper use of LangGraph
- Clean state management
- Error handling

## Submission Guidelines

### What to Submit:
- [ ] Completed `assignment.py` with working agent
- [ ] Test results on provided sample tasks
- [ ] Documentation of your agent's reasoning process
- [ ] Any custom tasks you created for testing
- [ ] Reflection on agent behavior and improvements

### Testing Your Agent:
```bash
# Test on all sample tasks
python test_agent.py

# Interactive mode
python assignment.py --interactive

# Verbose mode to see reasoning trace
python assignment.py --task math --verbose
```

## Advanced Concepts (Bonus)

### 1. Planning with Backtracking:
```python
def plan_with_backtracking(state):
    if current_plan_failed(state):
        return generate_alternative_plan(state)
    return continue_current_plan(state)
```

### 2. Agent Memory:
```python
class AgentMemory:
    def __init__(self):
        self.episodic_memory = []  # Past experiences
        self.semantic_memory = {}  # General knowledge
        
    def learn_from_experience(self, task, solution, outcome):
        # Store successful strategies
        pass
```

### 3. Multi-Agent Collaboration:
```python
class CollaborativeAgents:
    def __init__(self):
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.critic = CriticAgent()
        
    def solve_collaboratively(self, problem):
        # Agents work together
        pass
```

## Common Challenges & Solutions

### Challenge: Agent gets stuck in loops
**Solution**: Add loop detection and alternative planning

### Challenge: Poor reasoning quality
**Solution**: Improve prompts and add reflection steps

### Challenge: Complex state management
**Solution**: Use LangGraph's built-in state handling

## Additional Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Agent Patterns](https://blog.langchain.dev/planning-agents/)
- [Multi-Agent Systems](https://python.langchain.com/docs/use_cases/agent_simulations)

## Need Help?

- Start with simple single-step reasoning
- Use the provided example agent as reference
- Test incrementally with different task types
- Focus on clear state transitions

Ready to build your reasoning agent! 🤖🧠 