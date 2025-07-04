# 12-Class GenAI Bootcamp Roadmap

*1 hour sessions, 3 times per week, 4 weeks total*

## **WEEK 1: Python & GenAI Foundations**

### **Class 1: Python for AI Development**

**Duration:** 60 minutes

**Focus:** Essential Python skills for AI applications

**Session Structure:**

- **Theory (15 min):** Python ecosystem for AI, key libraries overview
- **Hands-on (35 min):**
    - Environment setup (Python, pip, virtual environments)
    - Working with requests, json, pandas
    - File handling and data manipulation
- **Wrap-up (10 min):** Q&A and assignment briefing

**Assignment:** Build a simple data processor that reads CSV files and outputs JSON summaries

### **Class 2: GenAI Fundamentals & APIs**

**Duration:** 60 minutes

**Focus:** Understanding LLMs and API integration

**Session Structure:**

- **Theory (20 min):** How LLMs work, API basics, prompt engineering principles
- **Hands-on (30 min):**
    - OpenAI API setup and first API call
    - Basic prompt engineering exercises
    - Error handling and rate limiting
- **Wrap-up (10 min):** Best practices discussion

**Assignment:** Create a prompt engineering playground that tests different prompting techniques

### **Class 3: Building Your First AI Application**

**Duration:** 60 minutes

**Focus:** Complete application development

**Session Structure:**

- **Theory (10 min):** Application architecture patterns
- **Hands-on (40 min):**
    - Build a command-line chatbot
    - Implement conversation memory
    - Add system prompts and personality
- **Wrap-up (10 min):** Demo and peer feedback

**Assignment:** Enhance chatbot with specific domain knowledge and custom commands

---

## **WEEK 2: RAG Systems & Document Processing**

### **Class 4: RAG Architecture & Vector Databases**

**Duration:** 60 minutes

**Focus:** Understanding retrieval-augmented generation

**Session Structure:**

- **Theory (25 min):** RAG concepts, embeddings, vector similarity, database options
- **Hands-on (25 min):**
    - Install and setup ChromaDB/FAISS
    - Generate embeddings with sentence-transformers
    - Basic similarity search implementation
- **Wrap-up (10 min):** Architecture review

**Assignment:** Build a simple document similarity finder using embeddings

### **Class 5: Building RAG with LangChain**

**Duration:** 60 minutes

**Focus:** LangChain framework for RAG

**Session Structure:**

- **Theory (15 min):** LangChain components, chains, and document loaders
- **Hands-on (35 min):**
    - Install LangChain and set up document loaders
    - Create text splitters and vector stores
    - Build basic RAG chain
- **Wrap-up (10 min):** Testing and debugging

**Assignment:** Create a RAG system that can answer questions about a specific document set

### **Class 6: Advanced RAG Techniques**

**Duration:** 60 minutes

**Focus:** Optimization and advanced patterns

**Session Structure:**

- **Theory (20 min):** Chunking strategies, metadata filtering, hybrid search
- **Hands-on (30 min):**
    - Implement different chunking methods
    - Add metadata filtering
    - Create evaluation metrics
- **Wrap-up (10 min):** Performance comparison

**Assignment:** Optimize RAG system performance and create evaluation benchmarks

---

## **WEEK 3: Agents & Workflow Automation**

### **Class 7: AI Agents & LangGraph Basics**

**Duration:** 60 minutes

**Focus:** Agent architecture and LangGraph introduction

**Session Structure:**

- **Theory (20 min):** Agent concepts, ReAct pattern, LangGraph overview
- **Hands-on (30 min):**
    - Install LangGraph
    - Create simple state graph
    - Build basic reasoning agent
- **Wrap-up (10 min):** Agent behavior analysis

**Assignment:** Build an agent that can perform multi-step reasoning tasks

### **Class 8: Function Calling & Tool Usage**

**Duration:** 60 minutes

**Focus:** Agents with external tools

**Session Structure:**

- **Theory (15 min):** Function calling, tool integration, safety considerations
- **Hands-on (35 min):**
    - Create custom tools (calculator, web search, file operations)
    - Implement function calling with OpenAI API
    - Build tool-using agent
- **Wrap-up (10 min):** Testing and error handling

**Assignment:** Create a research assistant agent with web search and summarization tools

### **Class 9: Visual Workflows with n8n**

**Duration:** 60 minutes

**Focus:** No-code/low-code automation

**Session Structure:**

- **Theory (15 min):** Workflow automation, n8n architecture
- **Hands-on (35 min):**
    - n8n setup and interface overview
    - Create AI-powered workflows
    - Connect APIs and triggers
- **Wrap-up (10 min):** Workflow sharing and collaboration

**Assignment:** Build an automated content generation workflow using n8n

---

## **WEEK 4: Production & Machine Learning**

### **Class 10: AI Assistants & Memory Management**

**Duration:** 60 minutes

**Focus:** Stateful AI applications

**Session Structure:**

- **Theory (15 min):** Conversation memory, context management, personalization
- **Hands-on (35 min):**
    - Implement conversation memory with LangChain
    - Build persistent storage for user preferences
    - Create context-aware responses
- **Wrap-up (10 min):** Memory optimization strategies

**Assignment:** Build a personal AI assistant with long-term memory and preferences

### **Class 11: Local Deployment & Model Optimization**

**Duration:** 60 minutes

**Focus:** Running models locally and deployment

**Session Structure:**

- **Theory (20 min):** Local vs cloud deployment, model quantization, hardware requirements
- **Hands-on (30 min):**
    - Setup Ollama for local LLM deployment
    - Model quantization techniques
    - Docker containerization basics
- **Wrap-up (10 min):** Deployment strategy discussion

**Assignment:** Deploy a local AI application using Docker and local models

### **Class 12: CNN Basics & Computer Vision**

**Duration:** 60 minutes

**Focus:** Introduction to deep learning and computer vision

**Session Structure:**

- **Theory (25 min):** CNN architecture, computer vision basics, transfer learning
- **Hands-on (25 min):**
    - Setup TensorFlow/PyTorch
    - Load pre-trained models
    - Build simple image classifier
- **Wrap-up (10 min):** Final project presentations and next steps

**Assignment:** Create a computer vision application that integrates with previous AI systems

---

## **Assignment Structure**

### **Weekly Themes:**

- **Week 1:** Foundation building - individual coding exercises
- **Week 2:** Document processing - RAG system development
- **Week 3:** Automation projects - agent and workflow creation
- **Week 4:** Integration capstone - combine all learned concepts

### **Final Project Options:**

1. **AI-Powered Research Assistant** - RAG + Agents + Web Search
2. **Automated Content Pipeline** - n8n + LLMs + Local Deployment
3. **Multimodal AI Application** - Text + Speech + Conversation Memory

### **Evaluation Criteria:**

- **Technical Implementation (40%)** - Code quality and functionality
- **Creativity & Problem-Solving (30%)** - Innovative use of concepts
- **Presentation & Documentation (20%)** - Clear explanation and demo
- **Collaboration & Peer Learning (10%)** - Helping others and participation

### **Required Tools & Setup:**

- Python 3.9+, Git, VS Code
- OpenAI / Gemini API key
- Google Colab account (backup for local issues)
- Discord/Slack for collaboration
- GitHub for project sharing
