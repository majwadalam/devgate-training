# Assignment 1: Python for AI Development

## Overview
Build a simple data processor that reads CSV files and outputs JSON summaries. This assignment focuses on essential Python skills for AI applications including file handling, data manipulation, and working with different data formats.

## Learning Objectives
- Set up Python virtual environments
- Work with CSV and JSON file formats
- Use pandas for data manipulation
- Implement basic data processing functions
- Handle file I/O operations

## Setup Instructions

### 1. Create Virtual Environment

#### On Windows:
```bash
# Navigate to assignment-1 directory
cd assignment-1

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
# Navigate to assignment-1 directory
cd assignment-1

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# For windows (Git Bash)
source venv/Scripts/activate

# For windows (Powershell)
./venv/Scripts/activate
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Check Python version
python --version

# Check installed packages
pip list
```

## Assignment Requirements

### Task Description
Create a Python script that:

1. **Reads CSV files** containing structured data
2. **Processes the data** to generate meaningful summaries
3. **Outputs JSON summaries** with key statistics and insights
4. **Handles multiple file formats** gracefully
5. **Includes error handling** for common issues

### Expected Features

#### Core Features (Required):
- [ ] Read CSV files using pandas
- [ ] Calculate basic statistics (count, mean, sum, etc.)
- [ ] Generate JSON output with summary data
- [ ] Handle missing or invalid data
- [ ] Process multiple CSV files

#### Bonus Features (Optional):
- [ ] Command-line interface for file selection
- [ ] Data visualization summaries
- [ ] Support for different CSV delimiters
- [ ] Automated file discovery in directories
- [ ] Data quality reporting

### Sample Input/Output

#### Input CSV (`sales_data.csv`):
```csv
date,product,category,price,quantity,customer_id
2024-01-01,Laptop,Electronics,999.99,2,CUST001
2024-01-02,Book,Education,29.99,1,CUST002
2024-01-03,Headphones,Electronics,149.99,3,CUST001
```

#### Expected JSON Output:
```json
{
  "file_name": "sales_data.csv",
  "summary": {
    "total_rows": 3,
    "total_revenue": 1329.96,
    "average_order_value": 443.32,
    "unique_customers": 2,
    "categories": {
      "Electronics": 2,
      "Education": 1
    },
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-03"
    }
  },
  "data_quality": {
    "missing_values": 0,
    "duplicate_rows": 0
  }
}
```

## Files Structure

```
assignment-1/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── example_script.py      # Working example solution
├── assignment.py          # Starter template for students
├── sample_data.csv        # Sample data for testing
└── venv/                  # Virtual environment (created by you)
```

## Getting Started

1. **Activate your virtual environment** (see setup instructions above)
2. **Run the example script** to see expected behavior:
   ```bash
   python example_script.py
   ```
3. **Examine the sample data** in `sample_data.csv`
4. **Complete the assignment** by editing `assignment.py`
5. **Test your solution** with the sample data

## Submission Guidelines

### What to Submit:
- [ ] Your completed `assignment.py` file
- [ ] Any additional CSV files you created for testing
- [ ] Generated JSON output files
- [ ] Brief documentation of your approach

### Testing Your Solution:
```bash
# Run your assignment script
python assignment.py

# Verify JSON output is valid
python -m json.tool output.json
```

## Common Issues & Solutions

### Virtual Environment Issues:
- **Problem**: `venv` command not found
- **Solution**: Use `python -m venv venv` instead

### Package Installation Issues:
- **Problem**: Permission denied during pip install
- **Solution**: Ensure virtual environment is activated

### File Path Issues:
- **Problem**: "File not found" errors
- **Solution**: Check file paths and current working directory

## Additional Resources

- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [JSON in Python](https://docs.python.org/3/library/json.html)
- [CSV File Processing](https://docs.python.org/3/library/csv.html)

## Need Help?

- Review the `example_script.py` for reference implementation
- Check the sample data format in `sample_data.csv`
- Ask questions during the wrap-up session
- Collaborate with peers on Discord/Slack

Good luck with your first AI development assignment! 🚀 