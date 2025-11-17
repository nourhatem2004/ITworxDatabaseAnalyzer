import os
from dotenv import load_dotenv
import google.generativeai as genai
import pyodbc
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import uuid
import warnings
import gradio as gr
warnings.filterwarnings("ignore")

load_dotenv() 

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 
qdrant = QdrantClient(":memory:")
llm = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_embedding(text: str):
    """Embed text using HuggingFace embeddings."""
    return embeddings.embed_query(text)

def connect_db():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'          
        'DATABASE=AdventureWorks2022;'       
        'Trusted_Connection=yes;'    
    )
    return conn

conn = connect_db()

def get_schema_info(conn):
    """Extract schema info (tables, columns, datatypes)."""
    query = """
    SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    ORDER BY TABLE_SCHEMA, TABLE_NAME;
    """
    cursor = conn.cursor()
    return cursor.execute(query).fetchall()

def create_collections():
    configs = [
        ("db_info_collection", "Database schema"),
    ]
    for cname, desc in configs:
        if not qdrant.collection_exists(cname):
            qdrant.create_collection(
                collection_name=cname,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE, on_disk=True)  # 384 for MiniLM
            )

create_collections()


def insert_schema_to_qdrant(schema_rows):
    points = []
    for row in schema_rows:
        schema, table, column, dtype = row
        text = f"Table {schema}.{table} has column {column} of type {dtype}"
        emb = get_embedding(text)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "schema": schema,
                "table": table,
                "column": column,
                "datatype": dtype,
                "description": text
            }
        ))
    qdrant.upsert(collection_name="db_info_collection", points=points)



def extract_words(question):
    prompt = f"""
You are an AI agent that extracts the 5 most relevant keywords from a user question in relevance to the chat history for use in a RAG system.

user question: {question}

Example (no history): 
User question: "Who are our highest earners in the company?" 
Output: ["salary", "earnings", "name", "HR", "employee"]

Rules:
- Always return exactly 5 keywords.
- Do not include explanatory text.
- Extract relevant words that might have corresponding table names, ignore words like "lowest", "sum" or other filtering words
- only output the keywords in list format like the example, output must be in this format and this format only: ["salary", "earnings", "name", "HR", "employee"]
"""
    keywords = llm.generate_content(prompt).text.strip("[]").replace('"', '').replace("'", "").split(",")

    keywords = [word.strip() for word in keywords]
    print(keywords)

    return keywords


def generate_sql_from_question(keywords,question):
    
    schema_context=''
    example_context=''
    for keyword in keywords:
        q_emb = get_embedding(keyword)

        schema_hits = qdrant.search(
            collection_name="db_info_collection",
            query_vector=q_emb,
            limit=5
        )

        schema_context += "\n".join([hit.payload["description"] for hit in schema_hits])

    print(schema_context)
    prompt = f"""
You are an expert SQL generator. 
Use the database schema and examples to answer.

Schema:
{schema_context}



Now generate SQL for the following question:
{question}


-Start with the query then explain your thought process, thought process should explain the type of data that will be retrieved with a brief explaination of it.
-start the query with "QUERY START:" and end it with "QUERY END, **NEVER** include ```sql``` tags"
-Only generate one interpretation, if you have more than one, choose the best or most effeccient one
IMPORTANT:*"You are generating SQL queries for Microsoft SQL Server (T-SQL). Always use SQL Server syntax, not MySQL or PostgreSQL.*
    """
    # print(schema_context)
    response = llm.generate_content(prompt)
    return response.text

def chat_prompt(data, thought_process,question):
    
    prompt = f"""
You are an expert Data Analyst. 
Use the following database results, and the thought process behind how they were retrieved to answer the user's question in a friendly way.

Database context:
{data}

Thought_process behind query that retrieved this data:
{thought_process}

Now generate response for the following question:
{question}

"""
    print(prompt)
    response = llm.generate_content(prompt)
    return response.text



def split_sql(sql):
    query = sql.split("QUERY START:")[1].split("QUERY END")[0].strip()
    thought_process = sql.split("QUERY END")[1]
    return query,thought_process

def retrieve_from_db(query, conn):
    s=""
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    for row in rows:
        s += str(row) + '\n'
    return s


def chat_func(message,history):
    schema = get_schema_info(conn)
    insert_schema_to_qdrant(schema)
    question = message
    keywords = extract_words(question)
    sql = generate_sql_from_question(keywords,question)
    query, thought_process = split_sql(sql)
    data = retrieve_from_db(query,conn)
    response = chat_prompt(data,thought_process,question)


    return response



if __name__ == "__main__":
    gr.ChatInterface(
    fn=chat_func, 
    type="messages"
    ).launch()
