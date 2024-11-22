#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
import json
from pprint import pprint
import pandas as pd
import boto3
from io import StringIO, BytesIO
import os
import numpy as np
from sqlalchemy import create_engine, text


# In[2]:


# In[3]:


CHOSEN_LITE_LLM_MODEL = 'Azure OpenAI GPT-4o (External)'

chat = ChatOpenAI(
    openai_api_base=litellm_proxy_endpoint, # set openai_api_base to the LiteLLM Proxy
    model = CHOSEN_LITE_LLM_MODEL,
    default_headers={'x-api-key': ''},
    temperature=temperature,
    api_key=bearer_token,
    streaming=False,
    user=bearer_token
)


# In[4]:


s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-1'
)

bucket_name = ""
target_folder = ""
target_files = ["portfolio.csv", "profile.csv", "transcript.csv"]
processed_dfs = []


# In[5]:


# Loop through each target file in the target_files list
for target_file in target_files:
    
    # Retrieve the file from S3 bucket and read its content into a pandas DataFrame
    response = s3.get_object(Bucket=bucket_name, Key=target_folder+target_file)
    csv_content = response['Body'].read().decode('utf-8')
    target_df = pd.read_csv(StringIO(csv_content))

    # Generate summary statistics for the entire dataset and sample data for examination
    data_desc = target_df.describe()  # Descriptive statistics of the dataset
    data_sample = (target_df.sample(n=50) if len(target_df) >= 50 else target_df)  # Sample 50 rows or use entire DataFrame
    data_head = target_df.head()  # First 5 rows of the dataset
    
    """
    Recognize column data types and generate code to cast to that data type. Also generate code to split a column into
    multiple columns if necessary.
    """
    # Try generating transformation code 5 times
    for _ in range(5):
        # Prepare the prompt to send to the AI model, including summary statistics and sample data
        prompt = f"You will receive a CSV with these summary statistics: \n{data_desc.to_csv()}\n" \
                 f"and this random sample of the dataset:\n{data_sample.to_csv()}\nDecide what data each column is storing, " \
                 f"decide the data type of each column in the CSV, and what the column name and data type should be to follow " \
                 f"the best data storage practices in a relational database. It may be necessary to split a column and create " \
                 f"a new column to allow for better data separation. Try to keep to primitive types as much as possible. " \
                 f"Return Python code to read the entire {target_file} file and transform it to the desired data types and columns. " \
                 f"Do not use sample data, read the entire CSV file and process that. The code should save the data as a CSV string " \
                 f"to a variable called 'code_output'."

        # Define system and human messages for AI model
        messages = [
            SystemMessage(content="You are an instruction-tuned large language model. Follow the user's instructions carefully. "
                                  "Respond using python code as plain text and nothing else. Do not use markdown. Your output should "
                                  "be able to be executed directly."),
            HumanMessage(content=prompt),
        ]

        try:
            # Call the AI model with the prompt to generate the transformation code
            chat_response = chat(messages).content
        except Exception as e:
            # Handle any exceptions during AI response generation
            print(str(e))
        execute_code = chat_response  # Store the AI's code response
        
        # Save the original CSV content to a file
        with open(target_file, "w") as fw:
            fw.write(csv_content)

        # Try executing the generated code
        namespace = {}
        try:
            exec(execute_code, namespace)
        except:
            # If execution fails, try again
            print("Exec failed, trying again")
            continue

        try:
            # Try reading the transformed DataFrame from the output of the executed code
            processed_df = pd.read_csv(StringIO(namespace["code_output"]))
            break  # Exit the loop if the execution was successful
        except:
            # If reading the transformed CSV fails, revert to the original DataFrame
            print("Couldn't convert back to dataframe, falling back to original dataframe")
            processed_df = target_df

    # Prepare the prompt to identify columns needing outlier removal or data validation
    prompt = f"You will receive a CSV with these summary statistics: \n{data_desc}\nand this sample of the dataset:" \
             f"\n{data_sample}\ndecide which columns should have outlier removal or data validation done. Columns storing " \
             "IDs or keys or foreign keys should be ignored. Return only a CSV of the column names that should have outlier " \
             "removal or data validation done and don't return anything else. If there are no columns where outlier removal " \
             "or data validation should be done, return 'None'."
    
    # Define system and human messages for AI model to detect columns needing validation or outlier removal
    messages = [
        SystemMessage(content="You are an instruction-tuned large language model. Follow the user's instructions carefully. "
                              "Respond using plain text CSV. Do not respond in markdown. Your output will be directly parsed as CSV. "
                              "Avoid repeating values in the CSV."),
        HumanMessage(content=prompt),
    ]

    try:
        # Call the AI model to determine columns needing validation or outlier removal
        chat_response = chat(messages).content
    except Exception as e:
        # Handle exceptions during AI response generation
        print(str(e))
    processing_cols = chat_response  # Capture the AI's response

    # List to store columns with anomalous outliers
    anomolous_outs = []
    
    # If there are columns requiring validation or outlier removal
    if processing_cols != "None":
        # Iterate through each column to examine the specified ones
        for col in processing_cols.split(","):
            col = col.strip()  # Clean up column name
            if col not in processed_df.columns:
                continue  # Skip if the column does not exist in the DataFrame
            
            # Retrieve the summary statistics and sample data for the column
            col_desc = processed_df[col].describe()
            col_head = processed_df[col].head()
            col_csv = (processed_df[col].sample(n=50000).to_csv() if len(processed_df[col]) > 50000 else processed_df[col].to_csv())

            # Create the prompt to detect anomalous values in the column
            prompt = f"You will receive a CSV with a single column titled {col} from a larger dataset. The head of the " \
                     f"dataset is:\n{data_head}\nThese are the summary statistics of the column: \n{col_desc}\n" \
                     f"and this head of the column:\n{col_head}\nFind anomalous values in this column. Find values that do not " \
                     "make sense given the column name, the value, and the frequency of that value. Return only a CSV " \
                     "of the anomalous values and nothing else. If there are no anomalous values, or if having nan " \
                     "values in this column would lose important information, return '[Response]\nNone'.\n" \
                     f"Data:\n{col_csv}"
            
            # Define system and human messages for AI to detect anomalies
            messages = [
                SystemMessage(content="You are an instruction-tuned large language model. Follow the user's instructions carefully. "
                                      "Reason your way through the user's question out loud and respond by printing '[Response]' "
                                      "then printing the result as a comma-separated list without headers or newlines. Avoid "
                                      "repeating values in the CSV."),
                HumanMessage(content=prompt),
            ]

            try:
                # Call the AI model to detect anomalous values in the column
                response = chat(messages)
                anomolous_outs.append((col, response))  # Store the anomalies for the column
            except Exception as e:
                # Handle any exceptions during AI response generation
                print(str(e))

    # Process anomalous values for columns that were flagged by the AI model
    for col, out in anomolous_outs:
        split_out = out.content.split("[Response]")
        if len(split_out) <= 1:
            continue  # Skip if no response was received
        
        o = split_out[1].strip()  # Get the anomalous values
        if o != "None":
            # Replace anomalous values with NaN in the DataFrame
            for val in o.split(","):
                val = val.strip()
                try:
                    processed_df.replace({col: type(processed_df.loc[0][col])(val)}, np.nan, inplace=True)
                except:
                    continue

    # Append the processed DataFrame to the list of processed DataFrames
    processed_dfs.append(processed_df)

    # Print a message indicating the file has been processed
    print(f"Done processing {target_file}")


# In[6]:


DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = ''
USER = ''
PASSWORD = ''
PORT = 5432
DATABASE = ''

engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")

for file in target_files:
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT conname
            FROM pg_constraint
            WHERE conrelid = '{file.split('.')[0]}'::regclass
              AND confrelid IS NOT NULL;
        """))
        constraints = result.fetchall()
        for constraint in constraints:
            conn.execute(text(f"ALTER TABLE IF EXISTS {file.split('.')[0]} DROP CONSTRAINT {constraint[0]} CASCADE"))
for file in target_files:
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {file.split('.')[0]} CASCADE"))
for i in range(len(processed_dfs)):
    
    processed_dfs[i].to_sql(target_files[i].split(".")[0], engine, if_exists='replace', index=False)


# In[7]:


samples = []
with engine.connect() as conn:
    tables_cursor = conn.execute(text("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public';"""))
    tables = tables_cursor.fetchall()
    
    table_info_cursor = conn.execute(text("""
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public';
    """))
    table_info = table_info_cursor.fetchall()

    for val in tables:
        table = val[0]
        
        samples.append(conn.execute(text(f"""
        SELECT * 
        FROM {table} 
        ORDER BY RANDOM() 
        LIMIT 50;
        """)))


# In[8]:


prompt = f"You will receive information about a PostgreSQL database which contains the following tables:\n{tables}\n"\
        f"The following gives (table_name, column_name, column_datatype) in that format for the database:\n{table_info}\n"
for i in range(len(tables)):
    prompt += f"Here is a random sample of the {tables[i][0]} table:\n{samples[i].fetchall()}\n"
prompt += "\nYour task is to analyze the columns based on their data type, column name, and value. You should then compare these"\
        " with the other columns and determine which columns should be a primary or foreign key for each table. Then, generate"\
        " SQL code to run in PostgreSQL to create these keys and relationships between the tables. "
messages = [
SystemMessage(content="You are a instruction-tuned large language model. Follow the user's instructions carefully. "\
                      "Reason your way through the user's question out loud and respond by printing '[Response]' "\
                      "then printing the result as a plain text SQL query. Do not use markdown."),        
HumanMessage(content=prompt),]
for i in range(10):
    try:
        sql_response = chat(messages)
    except Exception as e:
        print(str(e))
        break
    try:
        with engine.connect() as conn:
            conn.execute(text(sql_response.content.split("[Response]")[1]))
        break
    except Exception as e:
        messages.append(HumanMessage(content=f"That returned exception {e}. Try again."))


# In[9]:


for i in range(len(target_files)):
    s3.upload_fileobj(BytesIO(processed_dfs[i].to_csv().encode('utf-8')), bucket_name, f"parsed_hackathon_files/{target_files[i]}")
    

