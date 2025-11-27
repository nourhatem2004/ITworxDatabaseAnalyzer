import os, warnings
import pyodbc
import json
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from dotenv import load_dotenv
import google.generativeai as genai
import gradio as gr

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool



conn_str = (
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=localhost;'          
    'DATABASE=AdventureWorks2022;'       
    'Trusted_Connection=yes;'
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

def query(sql, params=None):
    try:
        cursor.execute(sql, params or [])
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    except Exception as e:
        print(f"SQL error Encountered: {e}")
        return f"SQL error Encountered: {e}"
    






    
result = query("""
    SELECT 
        t.TABLE_SCHEMA,
        t.TABLE_NAME,
        t.TABLE_TYPE,
        STRING_AGG(c.COLUMN_NAME, ', ') AS COLUMN_NAMES
    FROM INFORMATION_SCHEMA.TABLES t
    LEFT JOIN INFORMATION_SCHEMA.COLUMNS c 
        ON t.TABLE_SCHEMA = c.TABLE_SCHEMA 
        AND t.TABLE_NAME = c.TABLE_NAME
    WHERE t.TABLE_TYPE = 'BASE TABLE'
    GROUP BY t.TABLE_SCHEMA, t.TABLE_NAME, t.TABLE_TYPE
    ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME;
""")


print(result)