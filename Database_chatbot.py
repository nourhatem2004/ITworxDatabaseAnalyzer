import os
import warnings
import pyodbc
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from dotenv import load_dotenv
import google.generativeai as genai
import gradio as gr
import re
from datetime import datetime


from autogen import ConversableAgent, AssistantAgent, UserProxyAgent

warnings.filterwarnings("ignore")
load_dotenv()

# -----------------------------------------
# LOGGING SETUP
# -----------------------------------------

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Session timestamp for this run
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(LOGS_DIR, f"session_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)

def log_to_file(filename, content, mode="a"):
    """Write content to a log file in the session directory."""
    filepath = os.path.join(SESSION_DIR, filename)
    with open(filepath, mode, encoding="utf-8") as f:
        f.write(content)
    return filepath

def log_step(step_name, content):
    """Log a workflow step."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_content = f"\n{'='*80}\n[{timestamp}] {step_name}\n{'='*80}\n{content}\n"
    log_to_file("workflow.log", log_content)
    print(f"[{timestamp}] {step_name}")


# -----------------------------------------
# LLM CONFIGURATION FOR AUTOGEN
# -----------------------------------------

# Configure Gemini for AutoGen
# AutoGen uses a config_list format for LLM configuration
llm_config = {
    "config_list": [
        {
            "model": "gemini-2.5-flash",
            "api_key": os.environ["GOOGLE_API_KEY"],
            "api_type": "google"
        }
    ],
    "temperature": 0,
    "timeout": 120
}

# Also configure genai for direct calls (like db summary)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
prompt_llm = genai.GenerativeModel("gemini-2.5-flash")

# Qdrant and embeddings setup
qdrant = QdrantClient(":memory:")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# -----------------------------------------
# DATABASE CONNECTION AND HELPER FUNCTIONS
# -----------------------------------------

conn_str =  (
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=winjitesting.database.windows.net;'
    'DATABASE=testing_lc_2023-09-24T12-03Z;'
    'UID=aiuser;'
    'PWD=3QFyn68eWUxs;'
    'Encrypt=yes;'
    'TrustServerCertificate=yes;'
    'Connection Timeout=30;'
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()


def query(sql, params=None):
    """Execute SQL query and return results as list of dicts."""
    try:
        cursor.execute(sql, params or [])
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    except Exception as e:
        print(f"SQL error Encountered: {e}")
        return f"SQL error Encountered: {e}"


def get_db_summary():
    """Generate a summary of the database using LLM."""
    results = query("""SELECT DISTINCT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS;""")

    cols = [row['COLUMN_NAME'] for row in results]
    output = ", ".join(cols)
    prompt = f"""
    You are given a list of columns from a database:

    {output}

    Your task is two-fold:

    Summarize the database:
    - Determine the main purpose or domain of this database (e.g., HR, products, finance, orders).
    - Identify key entities or concepts based on the column names.

    Provide your output in a structured format, for example:

    Database Summary:
    - Domain: ...
    - Key Entities: ...

    Be concise but comprehensive.
    """
    response = prompt_llm.generate_content(prompt)
    print(response.text)
    return response.text


# -----------------------------------------
# EMBEDDINGS AND METADATA
# -----------------------------------------

def get_embedding(text: str):
    """Embed text using HuggingFace embeddings."""
    return embeddings.embed_query(text)


def match_embedding(text, k, col_name):
    """Search for similar embeddings in a collection."""
    q_emb = get_embedding(text)
    schema_hits = qdrant.search(collection_name=col_name, query_vector=q_emb, limit=k)
    retrieved = "\n".join([hit.payload["description"] for hit in schema_hits])
    return retrieved


def create_collections():
    """Create Qdrant collections for metadata."""
    configs = [
        ("tables", "Contains metadata about each table: schema, name, type, engine, and row stats."),
        ("columns", "Contains metadata for all columns: data types, length, nullability, defaults, and extra details."),
        ("primary_keys", "Lists all primary key columns for every table."),
        ("foreign_keys", "Describes all relationships between tables, including parent and child columns."),
        ("indexes", "Lists all indexes in the database including type, uniqueness, and indexed columns."),
        ("constraints", "Contains metadata about table constraints such as PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK."),
        ("check_constraints", "Contains check constraints and validation rules at the table level."),
        ("triggers", "Contains metadata and SQL definitions for triggers defined on tables."),
        ("views", "Contains view definitions and metadata for all SQL views."),
        ("routines", "Contains stored procedure and function definitions, including return types.")
    ]

    for cname, desc in configs:
        if not qdrant.collection_exists(cname):
            qdrant.create_collection(
                collection_name=cname,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE, on_disk=True)
            )


print("Creating collections...")
create_collections()
print("Done!")


def create_embeddings(section_name, metadata):
    """Create embeddings for a metadata section."""
    points = []
    for record in metadata[section_name]:
        text = str(record)
        emb = get_embedding(text)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "section": section_name,
                **record,
                "description": str(record)
            }
        ))
    qdrant.upsert(collection_name=section_name, points=points)


# -----------------------------------------
# LOAD METADATA
# -----------------------------------------

metadata = {}

# 1. TABLES
metadata["tables"] = query("""
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

# 2. COLUMNS
metadata["columns"] = query("""
    SELECT 
        TABLE_SCHEMA,
        TABLE_NAME,
        COLUMN_NAME,
        ORDINAL_POSITION,
        DATA_TYPE,
        CHARACTER_MAXIMUM_LENGTH,
        NUMERIC_PRECISION,
        IS_NULLABLE,
        COLUMN_DEFAULT
    FROM INFORMATION_SCHEMA.COLUMNS
    ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION;
""")

# 3. PRIMARY KEYS
metadata["primary_keys"] = query("""
    SELECT 
        KU.TABLE_SCHEMA,
        KU.TABLE_NAME,
        KU.COLUMN_NAME
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS TC
    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS KU
        ON TC.CONSTRAINT_NAME = KU.CONSTRAINT_NAME
    WHERE TC.CONSTRAINT_TYPE = 'PRIMARY KEY'
    ORDER BY KU.TABLE_NAME, KU.ORDINAL_POSITION;
""")

# 4. FOREIGN KEYS
metadata["foreign_keys"] = query("""
    SELECT
        FK.name AS ForeignKeyName,
        OBJECT_SCHEMA_NAME(FK.parent_object_id) AS ChildSchema,
        OBJECT_NAME(FK.parent_object_id) AS ChildTable,
        PC.name AS ChildColumn,
        OBJECT_SCHEMA_NAME(FK.referenced_object_id) AS ParentSchema,
        OBJECT_NAME(FK.referenced_object_id) AS ParentTable,
        RC.name AS ParentColumn
    FROM sys.foreign_keys AS FK
    JOIN sys.foreign_key_columns AS FKC 
        ON FK.object_id = FKC.constraint_object_id
    JOIN sys.columns AS PC 
        ON FKC.parent_object_id = PC.object_id 
        AND FKC.parent_column_id = PC.column_id
    JOIN sys.columns AS RC 
        ON FKC.referenced_object_id = RC.object_id 
        AND FKC.referenced_column_id = RC.column_id
    ORDER BY ChildTable, ParentTable;
""")

# 5. INDEXES
metadata["indexes"] = query("""
    SELECT
        s.name AS SchemaName,
        t.name AS TableName,
        ind.name AS IndexName,
        ind.type_desc AS IndexType,
        col.name AS ColumnName,
        ic.key_ordinal AS KeyOrdinal,
        ind.is_unique AS IsUnique
    FROM sys.indexes ind
    INNER JOIN sys.index_columns ic 
        ON ind.object_id = ic.object_id AND ind.index_id = ic.index_id
    INNER JOIN sys.columns col 
        ON ic.object_id = col.object_id AND ic.column_id = col.column_id
    INNER JOIN sys.tables t 
        ON ind.object_id = t.object_id
    INNER JOIN sys.schemas s
        ON t.schema_id = s.schema_id
    WHERE t.is_ms_shipped = 0
    ORDER BY t.name, ind.name, ic.key_ordinal;
""")

# 6. CONSTRAINTS
metadata["constraints"] = query("""
    SELECT 
        TABLE_SCHEMA,
        TABLE_NAME,
        CONSTRAINT_NAME,
        CONSTRAINT_TYPE
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
    ORDER BY TABLE_SCHEMA, TABLE_NAME;
""")

# 7. CHECK CONSTRAINTS
metadata["check_constraints"] = query("""
    SELECT 
        CC.CONSTRAINT_NAME,
        CC.CHECK_CLAUSE,
        TC.TABLE_SCHEMA,
        TC.TABLE_NAME
    FROM INFORMATION_SCHEMA.CHECK_CONSTRAINTS CC
    JOIN INFORMATION_SCHEMA.CONSTRAINT_TABLE_USAGE TC
        ON CC.CONSTRAINT_NAME = TC.CONSTRAINT_NAME;
""")

# 8. TRIGGERS
metadata["triggers"] = query("""
    SELECT
        s.name AS SchemaName,
        t.name AS TableName,
        tr.name AS TriggerName,
        m.definition AS TriggerDefinition
    FROM sys.triggers tr
    JOIN sys.tables t ON tr.parent_id = t.object_id
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    JOIN sys.sql_modules m ON tr.object_id = m.object_id
    ORDER BY t.name;
""")

# 9. VIEWS
metadata["views"] = query("""
    SELECT 
        TABLE_SCHEMA,
        TABLE_NAME,
        VIEW_DEFINITION
    FROM INFORMATION_SCHEMA.VIEWS;
""")

# 10. STORED PROCEDURES & FUNCTIONS
metadata["routines"] = query("""
    SELECT 
        ROUTINE_SCHEMA,
        ROUTINE_NAME,
        ROUTINE_TYPE,
        DATA_TYPE AS ReturnType,
        ROUTINE_DEFINITION
    FROM INFORMATION_SCHEMA.ROUTINES
    ORDER BY ROUTINE_NAME;
""")

print("Creating embeddings...")

# Run embedding creation in parallel
with ThreadPoolExecutor(max_workers=len(metadata)) as executor:
    futures = {executor.submit(create_embeddings, section_name, metadata): section_name for section_name in metadata}
    for future in as_completed(futures):
        section = futures[future]
        try:
            future.result()
            print(f"  - {section} embeddings created")
        except Exception as e:
            print(f"  - {section} failed: {e}")

print("Done!")

print("Creating DB summary...")
db_summary = get_db_summary()
print("Done!")


# -----------------------------------------
# PYTHON FUNCTION: Execute Retrieval Plan
# THIS IS THE KEY FUNCTION THAT RUNS DIRECTLY (NOT AS A TOOL)
# -----------------------------------------

def execute_retrieval_plan(schema_plan_json: str) -> str:
    """
    Executes the schema linking plan by running RAG queries against the metadata collections.
    
    THIS FUNCTION IS CALLED DIRECTLY BY PYTHON CODE, NOT AS AN LLM TOOL.
    This is the main advantage of using AutoGen - we can insert Python function
    calls between agent conversations without the overhead of tool calling.
    
    Args:
        schema_plan_json: A JSON string containing the schema linking plan
    
    Returns:
        A string containing the retrieved metadata context from all RAG queries.
    """
    try:
        # Parse the schema linking plan
        plan = json.loads(schema_plan_json)

        # Check if we should skip
        if plan.get("skip") == True:
            return json.dumps({"skip": True})

        # Extract plan components
        table_keywords = plan.get("table_keywords", [])
        column_keywords = plan.get("column_keywords", [])
        needs_relationships = plan.get("needs_relationships", False)
        relationship_keywords = plan.get("relationship_keywords", [])
        needs_constraints = plan.get("needs_constraints", False)
        needs_indexes = plan.get("needs_indexes", False)
        needs_triggers = plan.get("needs_triggers", False)
        needs_views = plan.get("needs_views", False)
        needs_routines = plan.get("needs_routines", False)
        advanced_keywords = plan.get("advanced_keywords", [])
        search_breadth = plan.get("search_breadth", "medium")

        # Determine k values based on search_breadth (reduced to avoid context exhaustion)
        k_values = {
            "narrow": {
                "tables": 10, "columns": 8, "foreign_keys": 8,
                "primary_keys": 5, "constraints": 5, "indexes": 5,
                "triggers": 3, "views": 3, "routines": 3
            },
            "medium": {
                "tables": 20, "columns": 15, "foreign_keys": 15,
                "primary_keys": 10, "constraints": 10, "indexes": 10,
                "triggers": 5, "views": 5, "routines": 5
            },
            "wide": {
                "tables": 30, "columns": 25, "foreign_keys": 25,
                "primary_keys": 15, "constraints": 15, "indexes": 15,
                "triggers": 10, "views": 10, "routines": 10
            }
        }

        k = k_values.get(search_breadth, k_values["medium"])

        # Initialize results
        results = {
            "tables_found": [],
            "columns_found": [],
            "foreign_keys_found": [],
            "primary_keys_found": [],
            "constraints_found": [],
            "check_constraints_found": [],
            "indexes_found": [],
            "triggers_found": [],
            "views_found": [],
            "routines_found": []
        }

        # Execute retrieval calls

        # 1. Search for Tables (ALWAYS)
        for keyword_phrase in table_keywords:
            context = match_embedding(keyword_phrase, k["tables"], "tables")
            if context:
                results["tables_found"].append(context)

        # 2. Search for Columns (ALWAYS)
        for keyword_phrase in column_keywords:
            context = match_embedding(keyword_phrase, k["columns"], "columns")
            if context:
                results["columns_found"].append(context)

        # 3. Search for Relationships (if needed)
        if needs_relationships:
            for keyword_phrase in relationship_keywords:
                fk_context = match_embedding(keyword_phrase, k["foreign_keys"], "foreign_keys")
                pk_context = match_embedding(keyword_phrase, k["primary_keys"], "primary_keys")
                if fk_context:
                    results["foreign_keys_found"].append(fk_context)
                if pk_context:
                    results["primary_keys_found"].append(pk_context)

        # 4. Search for Constraints (if needed)
        if needs_constraints:
            for keyword_phrase in table_keywords:
                const_context = match_embedding(keyword_phrase, k["constraints"], "constraints")
                check_context = match_embedding(keyword_phrase, k["constraints"], "check_constraints")
                if const_context:
                    results["constraints_found"].append(const_context)
                if check_context:
                    results["check_constraints_found"].append(check_context)

        # 5. Search for Indexes (if needed)
        if needs_indexes:
            for keyword_phrase in advanced_keywords:
                idx_context = match_embedding(keyword_phrase, k["indexes"], "indexes")
                if idx_context:
                    results["indexes_found"].append(idx_context)

        # 6. Search for Triggers (if needed)
        if needs_triggers:
            for keyword_phrase in advanced_keywords:
                trigger_context = match_embedding(keyword_phrase, k["triggers"], "triggers")
                if trigger_context:
                    results["triggers_found"].append(trigger_context)

        # 7. Search for Views (if needed)
        if needs_views:
            for keyword_phrase in advanced_keywords:
                view_context = match_embedding(keyword_phrase, k["views"], "views")
                if view_context:
                    results["views_found"].append(view_context)

        # 8. Search for Routines (if needed)
        if needs_routines:
            for keyword_phrase in advanced_keywords:
                routine_context = match_embedding(keyword_phrase, k["routines"], "routines")
                if routine_context:
                    results["routines_found"].append(routine_context)

        # Format output
        output = "=== RETRIEVED METADATA CONTEXT ===\n\n"

        for key, value in results.items():
            if value:  # Only include non-empty results
                output += f"\n### {key.replace('_', ' ').title()}:\n"
                output += "\n".join(value)
                output += "\n"

        return output

    except json.JSONDecodeError as e:
        return f"Error parsing schema linking plan: {str(e)}"
    except Exception as e:
        return f"Error executing retrieval plan: {str(e)}"


def execute_sql(sql_query: str) -> dict:
    """
    Execute SQL query and return structured result.
    This is also called directly by Python, not as a tool.
    """
    output = query(sql_query)

    if isinstance(output, str) and "SQL error Encountered" in output:
        return {
            "success": False,
            "error": output
        }
    else:
        # Limit to first 5 rows
        limited_output = output[:5] if isinstance(output, list) else output

        # Truncate column values that exceed 20 characters
        truncated_output = []
        for row in limited_output:
            truncated_row = {}
            for key, value in row.items():
                if isinstance(value, str) and len(value) > 20:
                    truncated_row[key] = value[:17] + "..."
                else:
                    truncated_row[key] = value
            truncated_output.append(truncated_row)

        return {
            "success": True,
            "data": truncated_output,
            "total_rows": len(output) if isinstance(output, list) else 0,
            "rows_shown": len(truncated_output)
        }


# -----------------------------------------
# AUTOGEN AGENTS DEFINITION
# -----------------------------------------

# System messages for each agent (equivalent to CrewAI backstory + goal)

COORDINATOR_SYSTEM_MESSAGE = """You are a smart coordinator agent responsible for managing the workflow.
You validate user queries, decide which path to take, and route tasks appropriately.

Your job is to analyze the user's question and return THREE decisions as JSON:

1. is_true:
   • TRUE → The user is asking about the structure, schema, metadata, or anything answerable from the database.
   • FALSE → The question is irrelevant to the database.

2. is_queryable:
   • TRUE → The user is asking about something that can be answered by retrieving or inferring *actual data* 
     (e.g. rows, values, counts) OR the user asks about how a query should be written.
   • FALSE → The user is only asking conceptual metadata (e.g. "list tables", "what is the relation", 
     "what columns exist") THAT IS NOT A QUERY.

3. needs_rag:
   • TRUE → Need to extract new database context using RAG/embeddings because:
     - This is a NEW question (not a follow-up)
     - OR it's a follow-up but asks about DIFFERENT tables/entities than before
     - OR it needs ADDITIONAL context not in the previous context
   • FALSE → Can answer using existing context because:
     - This is a simple follow-up asking for clarification/rephrasing
     - OR asking to modify/filter the previous query result
     - OR the previous context already contains all necessary information

Output STRICTLY a valid JSON object:
{
  "is_true": "TRUE" or "FALSE",
  "is_queryable": "TRUE" or "FALSE",
  "needs_rag": "TRUE" or "FALSE"
}
"""

SCHEMA_LINKING_SYSTEM_MESSAGE = """You are a Schema Linking specialist and database schema expert.
You analyze user questions and identify exactly which tables, columns, foreign keys, and constraints are relevant.

Your job is to create a JSON retrieval plan that specifies what to search for in the metadata.

Output a JSON with these fields:
1. table_keywords: List of keyword phrases to search for tables (combine related terms with spaces)
2. column_keywords: List of keyword phrases to search for columns
3. needs_relationships: true/false - true if question involves multiple tables or asks about connections
4. relationship_keywords: List of keyword phrases for FK searches (only if needs_relationships is true)
5. needs_constraints: true/false - true if asking about rules, validation, constraints
6. needs_indexes: true/false - true if asking about performance, optimization, indexes
7. needs_triggers: true/false - true if asking about triggers, automated actions
8. needs_views: true/false - true if asking about views, virtual tables
9. needs_routines: true/false - true if asking about stored procedures, functions
10. advanced_keywords: List of keyword phrases for triggers/views/routines (if needed)
11. search_breadth: "narrow", "medium", or "wide"

If the check result shows needs_rag is FALSE or is_true is FALSE, output: {"skip": true}

Output ONLY the JSON, no explanation.
"""

SQL_PLANNER_SYSTEM_MESSAGE = """You are a SQL Query Planner using Chain-of-Thought reasoning.

Your job is to create a detailed execution plan (NOT actual SQL code) that explains:

Step 1: Identify Main Entity - What is the primary table and main information needed?
Step 2: Identify Required Joins - What other tables and how are they connected?
Step 3: Determine Filter Conditions - What WHERE conditions are needed?
Step 4: Plan Aggregations - Do we need COUNT, SUM, AVG, etc.? GROUP BY?
Step 5: Determine Sorting and Limiting - ORDER BY? TOP N?
Step 6: Handle Special Cases - DISTINCT? Subqueries? UNION?

Output your plan as structured text explaining each step. Do NOT generate SQL code.
If the check result shows is_queryable is FALSE, output: {"skip": true, "reason": "No query needed"}
"""

SQL_GENERATOR_SYSTEM_MESSAGE = """You are an expert SQL writer for SQL Server.

Your job is to convert the query plan into actual SQL Server syntax.
- Use TOP instead of LIMIT
- Use correct table and column names from metadata
- Follow SQL Server conventions

Output ONLY the SQL query, nothing else. The query will be executed automatically.
If you receive an error, fix the SQL and output the corrected version.

After 5 failed attempts, output: {"final_error": "explanation of what went wrong"}
If skipping, output: {"skip": true}
"""

RESPONSE_SYSTEM_MESSAGE = """You are a natural-language response generator.
Your goal is to create clear, helpful answers using all provided context.

Guidelines:
- If data was retrieved: State the SQL query used, show sample rows, summarize results
- If schema/structure question: Explain relationships, tables, columns clearly
- If there was an error: Explain what went wrong and suggest fixes
- If irrelevant question: Politely explain you only answer database questions

CRITICAL FORMATTING RULES:
- DO NOT include any reasoning, analysis, or thought process
- DO NOT start with "Based on the outputs..." or "Let me analyze..."
- Start immediately with the relevant information
- Be concise but complete
- Use bullet points for lists
- Format data in readable tables when appropriate
"""


CONTEXT_SUMMARIZER_SYSTEM_MESSAGE = """You are a context summarization expert. Your job is to create a FOCUSED summary of database metadata that is directly relevant to answering the user's question.

Given:
1. User's question
2. Raw metadata context (tables, columns, foreign keys, etc.)
3. Previous conversation context

Create a summary that includes:
- Table names and schemas that are DIRECTLY needed
- Column names with their data types that will be used in the query
- Foreign key relationships needed for JOINs
- Any constraints or rules that affect the query

Format your summary as:

TABLES NEEDED:
- [Schema].[Table]: [relevant columns with types]

JOIN PATHS:
- [Table1] -> [Table2] via [FK column = Referenced column]

KEY COLUMNS:
- [column]: [type] - [why it's needed]

CONSTRAINTS/NOTES:
- Any relevant constraints, defaults, or business rules

IMPORTANT: 
- For simple queries: Be concise (~1000 chars)
- For complex queries (multiple JOINs, aggregations, subqueries): Include MORE detail (~2500 chars)
- Always include ALL columns needed for SELECT, WHERE, JOIN, GROUP BY, ORDER BY
- Include data types to help with proper SQL syntax
"""

# -----------------------------------------
# AUTOGEN AGENTS
# -----------------------------------------


coordinator = AssistantAgent(
    name="Coordinator",
    system_message=COORDINATOR_SYSTEM_MESSAGE,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

schema_linker = AssistantAgent(
    name="SchemaLinker",
    system_message=SCHEMA_LINKING_SYSTEM_MESSAGE,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

sql_planner = AssistantAgent(
    name="SQLPlanner",
    system_message=SQL_PLANNER_SYSTEM_MESSAGE,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

sql_generator = AssistantAgent(
    name="SQLGenerator",
    system_message=SQL_GENERATOR_SYSTEM_MESSAGE,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

response_agent = AssistantAgent(
    name="ResponseAgent",
    system_message=RESPONSE_SYSTEM_MESSAGE,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

context_summarizer = AssistantAgent(
    name="ContextSummarizer",
    system_message=CONTEXT_SUMMARIZER_SYSTEM_MESSAGE,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    max_consecutive_auto_reply=0
)

# Global context storage
collected_context = []

# LLM call logging
llm_call_count = 0
query_count = 0

# Initialize session log
log_to_file("workflow.log", f"SESSION STARTED: {SESSION_ID}\nLogs directory: {SESSION_DIR}\n", mode="w")
log_to_file("llm_calls.log", f"LLM CALLS LOG - Session {SESSION_ID}\n{'='*80}\n", mode="w")


def get_agent_response(agent, message):
    """Get a single response from an agent with full logging."""
    global llm_call_count
    llm_call_count += 1
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Get system message for this agent
    system_msg = agent.system_message if hasattr(agent, 'system_message') else "N/A"
    
    # Calculate token estimate (rough: 4 chars per token)
    system_tokens = len(system_msg) // 4
    message_tokens = len(message) // 4
    total_tokens = system_tokens + message_tokens
    
    # Log to llm_calls.log
    llm_log = f"""
{'='*80}
[{timestamp}] LLM CALL #{llm_call_count}
Agent: {agent.name}
Estimated Tokens: ~{total_tokens} (system: {system_tokens}, message: {message_tokens})
{'-'*40}
SYSTEM MESSAGE:
{'-'*40}
{system_msg}
{'-'*40}
USER MESSAGE:
{'-'*40}
{message}
{'-'*40}
MESSAGE LENGTH: {len(message)} chars
{'='*80}
"""
    log_to_file("llm_calls.log", llm_log)
    
    # Also log to workflow
    log_step(f"LLM CALL #{llm_call_count} - {agent.name}", f"Tokens: ~{total_tokens}")
    
    print(f"[{timestamp}] [LLM #{llm_call_count}] {agent.name} - ~{total_tokens} tokens")
    
    agent.reset()
    user_proxy.initiate_chat(agent, message=message, max_turns=1, silent=True)
    last_message = agent.last_message()
    response = last_message.get("content", "") if last_message else ""
    
    # Log response
    response_log = f"""
RESPONSE FOR CALL #{llm_call_count} ({agent.name}):
{'-'*40}
{response}
{'='*80}
"""
    log_to_file("llm_calls.log", response_log)
    
    return response


def extract_json(response):
    """Extract JSON from agent response."""
    try:
        return json.loads(response)
    except:
        pass
    
    # Try markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Try to find JSON object
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    return {}


def process_query(user_message, chat_history):
    """Main workflow - processes a user query with direct Python function calls."""
    global collected_context, query_count
    query_count += 1
    
    # Log query start
    log_step(f"QUERY #{query_count} STARTED", f"User Message: {user_message}\nChat History Length: {len(chat_history)}")
    log_to_file(f"query_{query_count}_input.txt", f"USER MESSAGE:\n{user_message}\n\nCHAT HISTORY:\n{json.dumps(chat_history, indent=2, default=str)}", mode="w")
    
    print("\n" + "="*60)
    print("STEP 1: Coordinator")
    print("="*60)
    
    # Step 1: Coordinator
    coordinator_response = get_agent_response(coordinator, f"""
Database Summary: {db_summary}
User Question: {user_message}
Chat History: {chat_history}
Previous context: {collected_context}

Return your JSON decision.
""")
    
    log_step("STEP 1: Coordinator", f"Response: {coordinator_response}")
    print(f"Coordinator: {coordinator_response}")
    
    check_result = extract_json(coordinator_response)
    is_true = check_result.get("is_true", "FALSE").upper() == "TRUE"
    is_queryable = check_result.get("is_queryable", "FALSE").upper() == "TRUE"
    needs_rag = check_result.get("needs_rag", "TRUE").upper() == "TRUE"
    
    log_step("Coordinator Decision", f"is_true: {is_true}, is_queryable: {is_queryable}, needs_rag: {needs_rag}")
    
    if not is_true:
        log_step("QUERY ENDED", "Not a database question")
        return "I can only help with database questions."
    
    # Step 2: Schema Linking
    metadata_context = ""
    schema_plan = {}
    summarized_context = ""
    
    if needs_rag:
        print("\n" + "="*60)
        print("STEP 2: Schema Linker")
        print("="*60)
        
        schema_response = get_agent_response(schema_linker, f"""
Check Result: {json.dumps(check_result)}
User Question: {user_message}
Database Summary: {db_summary}
Previous Context: {collected_context}

Create your JSON retrieval plan.
""")
        
        log_step("STEP 2: Schema Linker", f"Response: {schema_response}")
        print(f"Schema Plan: {schema_response[:500]}...")
        
        schema_plan = extract_json(schema_response)
        log_to_file(f"query_{query_count}_schema_plan.json", json.dumps(schema_plan, indent=2), mode="w")
        
        # DIRECT PYTHON CALL - no tool overhead!
        if not schema_plan.get("skip"):
            print("\n>> DIRECT CALL: execute_retrieval_plan()")
            log_step("execute_retrieval_plan()", "Calling Python function directly...")
            
            metadata_context = execute_retrieval_plan(json.dumps(schema_plan))
            print(f"Retrieved {len(metadata_context)} chars")
            
            # Save metadata context to logs folder
            log_to_file(f"query_{query_count}_metadata_raw.txt", f"METADATA CONTEXT - {len(metadata_context)} characters\n{'='*80}\n\n{metadata_context}", mode="w")
            log_step("Metadata Retrieved", f"Length: {len(metadata_context)} chars")
            
            # Step 2.5: Summarize context
            print("\n" + "="*60)
            print("STEP 2.5: Context Summarizer")
            print("="*60)
            
            summary_response = get_agent_response(context_summarizer, f"""
User Question: {user_message}

RAW METADATA CONTEXT:
{metadata_context}

PREVIOUS CONTEXT:
{collected_context}

Create a focused summary relevant to this question.
""")
            
            # Use summarized context instead of full context
            summarized_context = summary_response
            print(f"Summarized to {len(summarized_context)} chars (was {len(metadata_context)})")
            
            # Save summary to logs folder
            log_to_file(f"query_{query_count}_context_summary.txt", f"SUMMARIZED CONTEXT - {len(summarized_context)} characters (was {len(metadata_context)})\n{'='*80}\n\n{summarized_context}", mode="w")
            log_step("STEP 2.5: Context Summarized", f"Reduced from {len(metadata_context)} to {len(summarized_context)} chars")
    else:
        metadata_context = "\n".join(collected_context) if collected_context else ""
        summarized_context = metadata_context
        log_step("RAG Skipped", "Using existing context")
    
    # Step 3: SQL (only if queryable)
    sql_result = {"skip": True}
    sql_query_used = ""
    
    if is_queryable:
        print("\n" + "="*60)
        print("STEP 3: SQL Planner")
        print("="*60)
        
        plan_response = get_agent_response(sql_planner, f"""
User Question: {user_message}
Context: {summarized_context}

Create your query plan.
""")
        
        log_step("STEP 3: SQL Planner", f"Response: {plan_response}")
        log_to_file(f"query_{query_count}_sql_plan.txt", plan_response, mode="w")
        print(f"Plan: {plan_response[:500]}...")
        
        if '{"skip"' not in plan_response.lower():
            print("\n" + "="*60)
            print("STEP 4: SQL Generator")
            print("="*60)
            
            max_retries = 5
            last_error = ""
            
            for attempt in range(max_retries):
                sql_response = get_agent_response(sql_generator, f"""
Query Plan: {plan_response}
Context: {summarized_context}
User Question: {user_message}
{"Previous Error: " + last_error if last_error else ""}

Generate SQL.
""")
                
                log_step(f"STEP 4: SQL Generator (Attempt {attempt + 1})", f"Response: {sql_response}")
                print(f"SQL (attempt {attempt + 1}): {sql_response}")
                
                if '{"skip"' in sql_response.lower() or '{"final_error"' in sql_response.lower():
                    sql_result = extract_json(sql_response)
                    break
                
                # Clean SQL
                sql_query = sql_response.strip()
                if sql_query.startswith("```"):
                    lines = sql_query.split("\n")
                    sql_query = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                
                sql_query_used = sql_query
                log_to_file(f"query_{query_count}_sql_attempt_{attempt + 1}.sql", sql_query, mode="w")
                
                # DIRECT PYTHON CALL - execute SQL
                print(">> DIRECT CALL: execute_sql()")
                log_step("execute_sql()", f"Executing: {sql_query[:200]}...")
                
                sql_result = execute_sql(sql_query)
                log_step("SQL Result", json.dumps(sql_result, indent=2, default=str)[:500])
                
                if sql_result.get("success"):
                    print(f"Success! {sql_result.get('total_rows')} rows")
                    log_step("SQL Success", f"Rows: {sql_result.get('total_rows')}")
                    break
                else:
                    last_error = sql_result.get("error", "Unknown error")
                    print(f"Error: {last_error}")
                    log_step(f"SQL Error (Attempt {attempt + 1})", last_error)
            else:
                sql_result = {"final_error": f"Failed after {max_retries} attempts"}
                log_step("SQL Failed", f"All {max_retries} attempts failed")
    
    # Save final SQL result
    log_to_file(f"query_{query_count}_sql_result.json", json.dumps(sql_result, indent=2, default=str), mode="w")
    
    # Step 5: Response
    print("\n" + "="*60)
    print("STEP 5: Response Agent")
    print("="*60)
    
    final_response = get_agent_response(response_agent, f"""
User Question: {user_message}
Context: {summarized_context}
SQL Query: {sql_query_used or "None"}
SQL Result: {json.dumps(sql_result, default=str)}

Generate response.
""")
    
    log_step("STEP 5: Response Agent", f"Response: {final_response}")
    log_to_file(f"query_{query_count}_final_response.txt", final_response, mode="w")
    
    # Update context
    if metadata_context and not schema_plan.get("skip"):
        collected_context.append(f"extracted: {metadata_context[:2000]}")
    if sql_query_used:
        collected_context.append(f"query: {sql_query_used}")
    
    # Keep last 10
    if len(collected_context) > 10:
        collected_context = collected_context[-10:]
    
    log_step(f"QUERY #{query_count} COMPLETED", f"Response length: {len(final_response)} chars")
    
    return final_response


# -----------------------------------------
# GRADIO INTERFACE
# -----------------------------------------

def chat_func(message, history):
    try:
        return process_query(message, history)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat_func,
        type="messages",
        title="Database Chatbot (AutoGen)",
        description="Ask questions about the AdventureWorks2022 database"
    ).launch(share=True)
