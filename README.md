---
title: Data Processing Pipeline Documentation
tags: [2024Hackathons]

---


# Data Processing Pipeline Documentation

## Introduction

This documentation provides an overview of the data processing pipeline designed to transform raw CSV files into a structured format suitable for storage in a relational database. The pipeline leverages a combination of AWS services, Python libraries, and a Large Language Model (LLM) to automate data transformation, validation, and storage tasks.

## Table of Contents

1. [Imports and Library Setup](#imports-and-library-setup)
2. [Configuration and Environment Variables](#configuration-and-environment-variables)
3. [Initializing the LLM](#initializing-the-llm)
4. [Setting Up AWS S3 Client](#setting-up-aws-s3-client)
5. [Data Retrieval from S3](#data-retrieval-from-s3)
6. [Data Processing with LLM](#data-processing-with-llm)
    - [Column Analysis and Transformation](#column-analysis-and-transformation)
    - [Outlier Detection and Removal](#outlier-detection-and-removal)
7. [Storing Data into PostgreSQL](#storing-data-into-postgresql)
8. [Setting Up Database Constraints](#setting-up-database-constraints)
9. [Data Sampling for Verification](#data-sampling-for-verification)
10. [Conclusion](#conclusion)

## Imports and Library Setup

The pipeline begins by importing necessary Python libraries:

```python
import os
import json
import pandas as pd
import numpy as np
import boto3
from io import StringIO, BytesIO
from sqlalchemy import create_engine, text
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
```

These libraries are used for:

- **os**: Accessing environment variables.
- **json**: Handling JSON data.
- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **boto3**: Interacting with AWS services.
- **StringIO, BytesIO**: In-memory file operations.
- **sqlalchemy**: Database operations.
- **langchain**: Interacting with the LLM.

## Configuration and Environment Variables

Environment variables and configuration settings are defined to authenticate and interact with various services:

```python
# LLM Proxy Endpoint
litellm_proxy_endpoint = os.environ.get(
    "litellm_proxy_endpoint",
    "https://api-llm.example.com"
)
temperature = 0

# Authentication tokens and API keys (placeholders used here)
bearer_token = 'YOUR_BEARER_TOKEN'
x_api_key = 'YOUR_X_API_KEY'

# LLM Model Selection
CHOSEN_LITE_LLM_MODEL = 'Azure OpenAI GPT-4o (External)'
```

**Note**: Sensitive information like API keys and tokens should be securely stored and not hard-coded.

## Initializing the LLM

An instance of the `ChatOpenAI` model is initialized to interact with the LLM:

```python
chat = ChatOpenAI(
    openai_api_base=litellm_proxy_endpoint,  # Set openai_api_base to the LiteLLM Proxy
    model=CHOSEN_LITE_LLM_MODEL,
    default_headers={'x-api-key': x_api_key},
    temperature=temperature,
    api_key=bearer_token,
    streaming=False,
    user=bearer_token
)
```

## Setting Up AWS S3 Client

A Boto3 S3 client is set up to interact with AWS S3 for data retrieval:

```python
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-1'
)
```

## Data Retrieval from S3

The pipeline reads CSV files from a specified S3 bucket:

```python
bucket_name = "your-bucket-name"
target_folder = "your-target-folder/"
target_files = ["portfolio.csv", "profile.csv", "transcript.csv"]
processed_dfs = []
```

Each file is read into a Pandas DataFrame:

```python
for target_file in target_files:
    response = s3.get_object(Bucket=bucket_name, Key=target_folder + target_file)
    csv_content = response['Body'].read().decode('utf-8')
    target_df = pd.read_csv(StringIO(csv_content))
```

## Data Processing with LLM

### Column Analysis and Transformation

For each DataFrame, the pipeline:

1. **Generates Summary Statistics and Samples**: Provides the LLM with descriptive statistics and a sample of the data.

   ```python
   data_desc = target_df.describe()
   data_sample = (target_df.sample(n=50) if len(target_df) >= 50 else target_df)
   data_head = target_df.head()
   ```

2. **LLM Prompt for Transformation Code**: Constructs a prompt asking the LLM to analyze column data types, suggest transformations, and generate Python code for these transformations.

   ```python
   prompt = f"You will receive a csv with these summary statistics: 
{data_desc.to_csv()}
"
   prompt += f"and this random sample of the dataset:
{data_sample.to_csv()}
"
   prompt += "Decide what data each column is storing, decide the data type of each column in the csv..."
   ```

3. **Executes the Generated Code**: Runs the Python code returned by the LLM to perform the data transformations.

### Outlier Detection and Removal

1. **Identifies Columns for Outlier Removal**: Prompts the LLM to decide which columns require outlier removal or data validation.

2. **Detects Anomalous Values**: For each identified column, the LLM is prompted to find anomalous values.

3. **Removes Anomalous Values**: Replaces detected anomalies with `NaN` in the DataFrame.

   ```python
   processed_df.replace({col: anomalous_value}, np.nan, inplace=True)
   ```

## Storing Data into PostgreSQL

Establishes a connection to a PostgreSQL database using SQLAlchemy:

```python
DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = 'your-database-endpoint'
USER = 'your-username'
PASSWORD = 'your-password'
PORT = 5432
DATABASE = 'your-database-name'

engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")
```

## Setting Up Database Constraints

Uses the LLM to generate SQL code for setting up primary keys and foreign keys.

1. **Gathers Table and Column Information**.

2. **Prompts the LLM to Generate SQL Code**.

3. **Executes the Generated SQL Code**.

## Data Sampling for Verification

Samples data from the database to verify that the data has been correctly processed and stored.

## Conclusion

This pipeline demonstrates an automated approach to data transformation and validation using a Large Language Model. By integrating AWS services, Python libraries, and the LLM, the pipeline efficiently processes raw data, identifies necessary transformations, and ensures that data is stored in a well-structured relational database format.