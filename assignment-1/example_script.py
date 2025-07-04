#!/usr/bin/env python3
"""
Assignment 1: Data Processor Example Script
Author: GenAI Bootcamp
Description: Example implementation of CSV to JSON data processor
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, List


class DataProcessor:
    """A simple data processor that reads CSV files and generates JSON summaries."""
    
    def __init__(self):
        self.supported_formats = ['.csv']
    
    def read_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV file and return a pandas DataFrame.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: For other file reading errors
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            print(f"✅ Successfully loaded {len(df)} rows from {file_path}")
            return df
        
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {str(e)}")
            raise
    
    def calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistics from the DataFrame.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Dictionary containing basic statistics
        """
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns)
        }
        
        # Numeric columns analysis
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            stats["numeric_columns"] = {}
            for col in numeric_columns:
                stats["numeric_columns"][col] = {
                    "mean": round(df[col].mean(), 2),
                    "sum": round(df[col].sum(), 2),
                    "min": round(df[col].min(), 2),
                    "max": round(df[col].max(), 2)
                }
        
        return stats
    
    def analyze_sales_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Specific analysis for sales data format.
        
        Args:
            df (pd.DataFrame): Sales data DataFrame
            
        Returns:
            Dict[str, Any]: Sales-specific analysis
        """
        analysis = {}
        
        # Revenue analysis (if price and quantity columns exist)
        if 'price' in df.columns and 'quantity' in df.columns:
            df['revenue'] = df['price'] * df['quantity']
            analysis["revenue_analysis"] = {
                "total_revenue": round(df['revenue'].sum(), 2),
                "average_order_value": round(df['revenue'].mean(), 2),
                "highest_sale": round(df['revenue'].max(), 2),
                "lowest_sale": round(df['revenue'].min(), 2)
            }
        
        # Category analysis
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            analysis["category_analysis"] = {
                "unique_categories": len(category_counts),
                "category_distribution": category_counts.to_dict()
            }
            
            if 'revenue' in df.columns:
                category_revenue = df.groupby('category')['revenue'].sum()
                analysis["category_analysis"]["revenue_by_category"] = {
                    cat: round(rev, 2) for cat, rev in category_revenue.items()
                }
        
        # Customer analysis
        if 'customer_id' in df.columns:
            unique_customers = df['customer_id'].nunique()
            customer_orders = df['customer_id'].value_counts()
            analysis["customer_analysis"] = {
                "unique_customers": unique_customers,
                "average_orders_per_customer": round(customer_orders.mean(), 2),
                "repeat_customers": len(customer_orders[customer_orders > 1])
            }
        
        # Date analysis
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                analysis["date_analysis"] = {
                    "date_range": {
                        "start": df['date'].min().strftime('%Y-%m-%d'),
                        "end": df['date'].max().strftime('%Y-%m-%d')
                    },
                    "total_days": (df['date'].max() - df['date'].min()).days + 1
                }
            except Exception as e:
                print(f"⚠️  Could not parse dates: {str(e)}")
        
        # Regional analysis
        if 'region' in df.columns:
            region_counts = df['region'].value_counts()
            analysis["regional_analysis"] = {
                "regions": region_counts.to_dict()
            }
            
            if 'revenue' in df.columns:
                region_revenue = df.groupby('region')['revenue'].sum()
                analysis["regional_analysis"]["revenue_by_region"] = {
                    region: round(rev, 2) for region, rev in region_revenue.items()
                }
        
        return analysis
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Data quality metrics
        """
        quality_metrics = {
            "missing_values": {
                "total": df.isnull().sum().sum(),
                "by_column": df.isnull().sum().to_dict()
            },
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.astype(str).to_dict()
        }
        
        return quality_metrics
    
    def generate_summary(self, file_path: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate complete summary of the data.
        
        Args:
            file_path (str): Original file path
            df (pd.DataFrame): Processed data
            
        Returns:
            Dict[str, Any]: Complete summary
        """
        summary = {
            "file_info": {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "processed_at": datetime.now().isoformat()
            },
            "basic_statistics": self.calculate_basic_stats(df),
            "sales_analysis": self.analyze_sales_data(df),
            "data_quality": self.check_data_quality(df)
        }
        
        return summary
    
    def save_json_summary(self, summary: Dict[str, Any], output_path: str) -> None:
        """
        Save summary to JSON file.
        
        Args:
            summary (Dict[str, Any]): Summary data
            output_path (str): Output file path
        """
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            converted_summary = convert_numpy_types(summary)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_summary, f, indent=2, ensure_ascii=False)
            print(f"✅ Summary saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving summary: {str(e)}")
            raise
    
    def process_file(self, input_file: str, output_file: str = None) -> Dict[str, Any]:
        """
        Process a single CSV file and generate JSON summary.
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str, optional): Path to output JSON file
            
        Returns:
            Dict[str, Any]: Generated summary
        """
        print(f"\n🔄 Processing file: {input_file}")
        
        # Read the CSV file
        df = self.read_csv_file(input_file)
        
        # Generate summary
        summary = self.generate_summary(input_file, df)
        
        # Save to JSON if output file specified
        if output_file:
            self.save_json_summary(summary, output_file)
        
        return summary
    
    def process_multiple_files(self, file_list: List[str], output_dir: str = "output") -> Dict[str, Any]:
        """
        Process multiple CSV files.
        
        Args:
            file_list (List[str]): List of CSV file paths
            output_dir (str): Directory to save output files
            
        Returns:
            Dict[str, Any]: Combined processing results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = {
            "processed_files": [],
            "errors": [],
            "summary": {
                "total_files": len(file_list),
                "successful": 0,
                "failed": 0
            }
        }
        
        for file_path in file_list:
            try:
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(output_dir, f"{file_name}_summary.json")
                
                summary = self.process_file(file_path, output_file)
                results["processed_files"].append({
                    "input_file": file_path,
                    "output_file": output_file,
                    "status": "success"
                })
                results["summary"]["successful"] += 1
                
            except Exception as e:
                error_info = {
                    "file": file_path,
                    "error": str(e)
                }
                results["errors"].append(error_info)
                results["summary"]["failed"] += 1
                print(f"❌ Failed to process {file_path}: {str(e)}")
        
        return results


def main():
    """Main function to demonstrate the data processor."""
    print("🚀 GenAI Bootcamp - Assignment 1: Data Processor")
    print("=" * 50)
    
    # Initialize the processor
    processor = DataProcessor()
    
    # Process the sample data
    sample_file = "sample_data.csv"
    
    if os.path.exists(sample_file):
        try:
            # Process single file
            summary = processor.process_file(sample_file, "sample_data_summary.json")
            
            # Display key results
            print("\n📊 Summary Results:")
            print("-" * 30)
            
            if "basic_statistics" in summary:
                stats = summary["basic_statistics"]
                print(f"📁 Total rows: {stats.get('total_rows', 'N/A')}")
                print(f"📁 Total columns: {stats.get('total_columns', 'N/A')}")
            
            if "sales_analysis" in summary and "revenue_analysis" in summary["sales_analysis"]:
                revenue = summary["sales_analysis"]["revenue_analysis"]
                print(f"💰 Total revenue: ${revenue.get('total_revenue', 'N/A')}")
                print(f"💰 Average order: ${revenue.get('average_order_value', 'N/A')}")
            
            if "sales_analysis" in summary and "customer_analysis" in summary["sales_analysis"]:
                customers = summary["sales_analysis"]["customer_analysis"]
                print(f"👥 Unique customers: {customers.get('unique_customers', 'N/A')}")
            
            print(f"\n✅ Complete summary saved to: sample_data_summary.json")
            print("\n💡 Try editing sample_data.csv and running the script again!")
            
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
    else:
        print(f"❌ Sample file '{sample_file}' not found!")
        print("💡 Make sure you're running this script from the assignment-1 directory")


if __name__ == "__main__":
    main() 