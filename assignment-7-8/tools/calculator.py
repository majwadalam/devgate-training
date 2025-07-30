"""
Simple calculator tool for AI agents
"""

import math
from typing import Union, Dict, Any

class Calculator:
    """Simple calculator with basic mathematical operations"""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a"""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent"""
        return base ** exponent
    
    def square_root(self, x: float) -> float:
        """Calculate square root"""
        if x < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(x)
    
    def percentage(self, value: float, percent: float) -> float:
        """Calculate percentage of a value"""
        return (percent / 100) * value
    
    def compound_interest(self, principal: float, rate: float, years: float, 
                         compounds_per_year: int = 1) -> float:
        """Calculate compound interest"""
        return principal * (1 + rate / compounds_per_year) ** (compounds_per_year * years)
    
    def average(self, numbers: list) -> float:
        """Calculate average of a list of numbers"""
        if not numbers:
            raise ValueError("Cannot calculate average of empty list")
        return sum(numbers) / len(numbers)
    
    def evaluate_expression(self, expression: str) -> Union[float, str]:
        """
        Safely evaluate a mathematical expression
        WARNING: This is a simplified version. In production, use a proper parser.
        """
        try:
            # Remove spaces and check for dangerous operations
            clean_expr = expression.replace(" ", "")
            
            # Basic safety check - only allow numbers, operators, and parentheses
            allowed_chars = set("0123456789+-*/().%")
            if not all(c in allowed_chars for c in clean_expr):
                return "Error: Invalid characters in expression"
            
            # Evaluate the expression
            result = eval(clean_expr)
            return float(result)
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"

# Tool functions for agent integration
def create_calculator_tools() -> Dict[str, Any]:
    """Create calculator tool functions for agent use"""
    calc = Calculator()
    
    return {
        "add": {
            "function": calc.add,
            "description": "Add two numbers",
            "parameters": ["a: float", "b: float"]
        },
        "subtract": {
            "function": calc.subtract,
            "description": "Subtract b from a",
            "parameters": ["a: float", "b: float"]
        },
        "multiply": {
            "function": calc.multiply,
            "description": "Multiply two numbers",
            "parameters": ["a: float", "b: float"]
        },
        "divide": {
            "function": calc.divide,
            "description": "Divide a by b",
            "parameters": ["a: float", "b: float"]
        },
        "percentage": {
            "function": calc.percentage,
            "description": "Calculate percentage of a value",
            "parameters": ["value: float", "percent: float"]
        },
        "compound_interest": {
            "function": calc.compound_interest,
            "description": "Calculate compound interest",
            "parameters": ["principal: float", "rate: float", "years: float", "compounds_per_year: int = 1"]
        },
        "average": {
            "function": calc.average,
            "description": "Calculate average of numbers",
            "parameters": ["numbers: list"]
        },
        "evaluate": {
            "function": calc.evaluate_expression,
            "description": "Evaluate a mathematical expression",
            "parameters": ["expression: str"]
        }
    }

if __name__ == "__main__":
    # Test the calculator
    calc = Calculator()
    
    print("Calculator Test:")
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 * 7 = {calc.multiply(6, 7)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")
    print(f"2^3 = {calc.power(2, 3)}")
    print(f"√16 = {calc.square_root(16)}")
    print(f"20% of 150 = {calc.percentage(150, 20)}")
    print(f"Average of [1,2,3,4,5] = {calc.average([1,2,3,4,5])}")
    print(f"Expression '2+3*4' = {calc.evaluate_expression('2+3*4')}")
    print(f"Compound interest $1000 at 5% for 3 years = {calc.compound_interest(1000, 0.05, 3)}") 