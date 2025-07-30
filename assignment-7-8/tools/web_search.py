"""
Web search simulator tool for AI agents
Note: This is a mock implementation for educational purposes.
In production, you would integrate with real search APIs.
"""

import random
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Represents a search result"""
    title: str
    url: str
    snippet: str
    relevance_score: float

class WebSearchSimulator:
    """Simulated web search tool for AI agents"""
    
    def __init__(self):
        """Initialize the web search simulator with mock data"""
        self.mock_results = {
            # AI and Technology
            "artificial intelligence": [
                SearchResult(
                    "What is Artificial Intelligence? | IBM",
                    "https://ibm.com/topics/artificial-intelligence",
                    "Artificial intelligence leverages computers and machines to mimic problem-solving and decision-making capabilities of the human mind.",
                    0.95
                ),
                SearchResult(
                    "AI Definition & Examples | McKinsey",
                    "https://mckinsey.com/featured-insights/mckinsey-explainers/what-is-ai",
                    "AI is a machine's ability to perform cognitive functions associated with human minds, such as perceiving, reasoning, learning...",
                    0.92
                ),
                SearchResult(
                    "Artificial Intelligence News | MIT Technology Review",
                    "https://technologyreview.com/topic/artificial-intelligence/",
                    "Latest news and breakthroughs in artificial intelligence research and applications.",
                    0.88
                )
            ],
            "machine learning": [
                SearchResult(
                    "Machine Learning Explained | AWS",
                    "https://aws.amazon.com/machine-learning/what-is-machine-learning/",
                    "Machine learning is a subset of artificial intelligence that gives systems the ability to automatically learn and improve.",
                    0.94
                ),
                SearchResult(
                    "Introduction to Machine Learning | Coursera",
                    "https://coursera.org/learn/machine-learning",
                    "Learn the fundamentals of machine learning with hands-on programming assignments.",
                    0.90
                ),
                SearchResult(
                    "Scikit-learn: Machine Learning in Python",
                    "https://scikit-learn.org/",
                    "Simple and efficient tools for predictive data analysis built on NumPy, SciPy, and matplotlib.",
                    0.87
                )
            ],
            "python programming": [
                SearchResult(
                    "Python.org - Official Python Website",
                    "https://python.org/",
                    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
                    0.96
                ),
                SearchResult(
                    "Learn Python Programming | Codecademy",
                    "https://codecademy.com/learn/learn-python-3",
                    "Learn Python 3 from scratch. No prior programming experience required.",
                    0.91
                ),
                SearchResult(
                    "Python Tutorial | W3Schools",
                    "https://w3schools.com/python/",
                    "Well organized and easy to understand Python tutorial with lots of examples.",
                    0.89
                )
            ],
            # Renewable Energy
            "solar energy": [
                SearchResult(
                    "Solar Energy Benefits | U.S. Department of Energy",
                    "https://energy.gov/eere/solar/solar-energy-united-states",
                    "Solar energy is the most abundant energy resource on earth. Learn about its benefits and applications.",
                    0.93
                ),
                SearchResult(
                    "How Solar Panels Work | National Geographic",
                    "https://nationalgeographic.com/science/article/solar-panels",
                    "Solar panels convert sunlight into electricity through photovoltaic cells.",
                    0.90
                ),
                SearchResult(
                    "Solar Power Cost Trends | IRENA",
                    "https://irena.org/solar-costs",
                    "Global weighted-average cost of electricity from solar PV declined 85% between 2010-2020.",
                    0.87
                )
            ],
            "wind energy": [
                SearchResult(
                    "Wind Power Facts | American Wind Energy Association",
                    "https://awea.org/wind-101",
                    "Wind power is one of the fastest-growing energy sources in the world.",
                    0.92
                ),
                SearchResult(
                    "How Wind Turbines Work | HowStuffWorks",
                    "https://science.howstuffworks.com/environmental/green-science/wind-power.htm",
                    "Wind turbines convert the kinetic energy in wind into mechanical power.",
                    0.89
                ),
                SearchResult(
                    "Offshore Wind Energy | NREL",
                    "https://nrel.gov/wind/offshore.html",
                    "Offshore wind resources are abundant, strong, and consistent.",
                    0.86
                )
            ],
            # Job Market
            "remote work": [
                SearchResult(
                    "State of Remote Work 2024 | Buffer",
                    "https://buffer.com/state-of-remote-work",
                    "99% of remote workers would like to work remotely, at least some of the time, for the rest of their careers.",
                    0.94
                ),
                SearchResult(
                    "Benefits of Remote Work | Harvard Business Review",
                    "https://hbr.org/2022/01/research-knowledge-workers-are-more-productive-from-home",
                    "Research shows knowledge workers are more productive from home.",
                    0.91
                ),
                SearchResult(
                    "Remote Work Statistics | Owl Labs",
                    "https://owllabs.com/state-of-remote-work/",
                    "Comprehensive statistics and trends about remote work adoption.",
                    0.88
                )
            ]
        }
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Simulate a web search for the given query
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        query_lower = query.lower()
        
        # Find the best matching mock results
        best_matches = []
        
        for key, results in self.mock_results.items():
            if key in query_lower or any(word in query_lower for word in key.split()):
                best_matches.extend(results)
        
        # If no specific matches, return some general results
        if not best_matches:
            best_matches = [
                SearchResult(
                    f"Search results for '{query}'",
                    f"https://example.com/search?q={query.replace(' ', '+')}",
                    f"Various web pages related to '{query}' with relevant information and resources.",
                    0.70
                ),
                SearchResult(
                    f"Wikipedia - {query}",
                    f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    f"Wikipedia article providing comprehensive information about {query}.",
                    0.75
                ),
                SearchResult(
                    f"Latest news about {query}",
                    f"https://news.google.com/search?q={query.replace(' ', '+')}",
                    f"Recent news articles and updates related to {query}.",
                    0.65
                )
            ]
        
        # Sort by relevance score and return top results
        best_matches.sort(key=lambda x: x.relevance_score, reverse=True)
        return best_matches[:max_results]
    
    def get_summary(self, query: str) -> str:
        """
        Get a summary of information about the query
        
        Args:
            query: Search query
            
        Returns:
            Summary string
        """
        results = self.search(query, max_results=3)
        
        if not results:
            return f"No specific information found for '{query}'. You may want to try a different search term."
        
        # Combine snippets to create a summary
        snippets = [result.snippet for result in results]
        summary = " ".join(snippets)
        
        return f"Summary for '{query}': {summary}"
    
    def fact_check(self, statement: str) -> Dict[str, Any]:
        """
        Simulate fact-checking a statement
        
        Args:
            statement: Statement to fact-check
            
        Returns:
            Dictionary with fact-check results
        """
        # This is a very simplified mock fact-checker
        # In reality, this would involve complex verification processes
        
        confidence_keywords = {
            "ai": ["artificial intelligence", "machine learning", "automation"],
            "python": ["programming", "language", "development"],
            "renewable": ["solar", "wind", "clean energy", "sustainable"],
            "remote": ["work", "telecommuting", "distributed teams"]
        }
        
        statement_lower = statement.lower()
        confidence = 0.5  # Default neutral confidence
        
        for category, keywords in confidence_keywords.items():
            if any(keyword in statement_lower for keyword in keywords):
                confidence = random.uniform(0.7, 0.9)
                break
        
        # Simulate verification result
        if confidence > 0.8:
            verification = "Likely accurate"
        elif confidence > 0.6:
            verification = "Partially verifiable"
        else:
            verification = "Requires further verification"
        
        return {
            "statement": statement,
            "verification": verification,
            "confidence": round(confidence, 2),
            "sources_found": len(self.search(statement, max_results=3)),
            "recommendation": "Cross-reference with multiple authoritative sources for important decisions."
        }

# Tool functions for agent integration
def create_web_search_tools() -> Dict[str, Any]:
    """Create web search tool functions for agent use"""
    searcher = WebSearchSimulator()
    
    return {
        "search": {
            "function": searcher.search,
            "description": "Search the web for information",
            "parameters": ["query: str", "max_results: int = 5"]
        },
        "get_summary": {
            "function": searcher.get_summary,
            "description": "Get a summary of information about a topic",
            "parameters": ["query: str"]
        },
        "fact_check": {
            "function": searcher.fact_check,
            "description": "Fact-check a statement",
            "parameters": ["statement: str"]
        }
    }

if __name__ == "__main__":
    # Test the web search simulator
    searcher = WebSearchSimulator()
    
    test_queries = [
        "artificial intelligence",
        "machine learning applications",
        "solar energy benefits",
        "remote work productivity"
    ]
    
    print("Web Search Simulator Test:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        print("-" * 30)
        
        results = searcher.search(query, max_results=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Snippet: {result.snippet}")
            print(f"   Relevance: {result.relevance_score}")
        
        print(f"\nSummary: {searcher.get_summary(query)}")
        print(f"Fact-check sample: {searcher.fact_check(f'{query} is important')}")
        print("=" * 50) 