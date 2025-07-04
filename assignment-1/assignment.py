#!/usr/bin/env python3
"""
Assignment 1: Data Processor - Student Template
Name: [Your Name Here]
Date: [Date]
Description: Complete this script to read CSV files and generate JSON summaries
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, List


class DataProcessor:
    """Complete this class to process CSV files and generate JSON summaries."""
    
    def __init__(self):
        """Initialize the DataProcessor."""
        # TODO: Add any initialization code you need
        pass
    
    def read_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV file and return a pandas DataFrame.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
            
        TODO: Implement this method
        - Check if file exists
        - Read CSV using pandas
        - Handle errors appropriately
        - Return the DataFrame
        """
        # Your code here
        pass
    
    def calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistics from the DataFrame.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Dictionary containing basic statistics
            
        TODO: Calculate and return:
        - Total number of rows
        - Total number of columns
        - Column names
        - For numeric columns: mean, sum, min, max
        """
        stats = {}
        
        # Your code here
        # Hint: Use len(df) for row count
        # Hint: Use df.select_dtypes(include=['number']) for numeric columns
        
        return stats
    
    def analyze_sales_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform sales-specific analysis.
        
        Args:
            df (pd.DataFrame): Sales data DataFrame
            
        Returns:
            Dict[str, Any]: Sales analysis results
            
        TODO: Implement analysis for:
        - Revenue calculation (price * quantity)
        - Category analysis
        - Customer analysis
        - Date range analysis (if date column exists)
        """
        analysis = {}
        
        # Revenue analysis
        # TODO: Check if 'price' and 'quantity' columns exist
        # TODO: Calculate total revenue, average order value
        
        # Category analysis
        # TODO: Count unique categories
        # TODO: Calculate revenue by category
        
        # Customer analysis
        # TODO: Count unique customers
        # TODO: Find repeat customers
        
        # Your code here
        
        return analysis
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Data quality metrics
            
        TODO: Check for:
        - Missing values (total and by column)
        - Duplicate rows
        - Data types of each column
        """
        quality_metrics = {}
        
        # Your code here
        # Hint: Use df.isnull().sum() for missing values
        # Hint: Use df.duplicated().sum() for duplicates
        
        return quality_metrics
    
    def generate_summary(self, file_path: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate complete summary of the data.
        
        Args:
            file_path (str): Original file path
            df (pd.DataFrame): Processed data
            
        Returns:
            Dict[str, Any]: Complete summary
            
        TODO: Combine all analysis methods into a single summary
        """
        summary = {
            "file_info": {
                "file_name": os.path.basename(file_path),
                "processed_at": datetime.now().isoformat()
            }
            # TODO: Add results from other analysis methods
        }
        
        # Your code here
        
        return summary
    
    def save_json_summary(self, summary: Dict[str, Any], output_path: str) -> None:
        """
        Save summary to JSON file.
        
        Args:
            summary (Dict[str, Any]): Summary data
            output_path (str): Output file path
            
        TODO: Save the summary dictionary as a JSON file
        """
        # Your code here
        # Hint: Use json.dump() with proper formatting
        pass
    
    def process_file(self, input_file: str, output_file: str = None) -> Dict[str, Any]:
        """
        Process a single CSV file and generate JSON summary.
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str, optional): Path to output JSON file
            
        Returns:
            Dict[str, Any]: Generated summary
            
        TODO: Orchestrate the entire processing pipeline
        """
        print(f"Processing file: {input_file}")
        
        # TODO: Read the CSV file
        # TODO: Generate summary
        # TODO: Save to JSON if output file specified
        # TODO: Return the summary
        
        pass


def main():
    """Main function to test your data processor."""
    print("Assignment 1: Data Processor")
    print("=" * 30)
    
    # TODO: Create an instance of DataProcessor
    processor = DataProcessor()
    
    # TODO: Process the sample_data.csv file
    sample_file = "sample_data.csv"
    
    if os.path.exists(sample_file):
        try:
            # TODO: Process the file and save summary
            summary = processor.process_file(sample_file, "my_summary.json")
            
            # TODO: Print some key results
            print("Processing completed successfully!")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print(f"Sample file '{sample_file}' not found!")


# Test your implementation
if __name__ == "__main__":
    main()


# BONUS CHALLENGES (Optional):
# 1. Add command-line argument support to specify input/output files
# 2. Create data visualizations using matplotlib
# 3. Support processing multiple files at once
# 4. Add data validation and cleaning features
# 5. Create a web interface using Flask or Streamlit


# TESTING CHECKLIST:
# □ Script runs without errors
# □ Reads CSV file successfully
# □ Calculates basic statistics correctly
# □ Generates valid JSON output
# □ Handles missing data appropriately
# □ Includes proper error handling
# □ Code is well-commented and readable


# SUBMISSION:
# 1. Complete all TODO sections
# 2. Test with sample_data.csv
# 3. Generate and verify JSON output
# 4. Add your name and date at the top
# 5. Submit assignment.py and generated JSON file 