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


warnings.filterwarnings("ignore")

load_dotenv() 


llm = LLM(
    model="gemini-2.5-flash",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.7
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 
prompt_llm = genai.GenerativeModel("gemini-2.5-flash")
qdrant = QdrantClient(":memory:")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")




# -----------------------------------------
# CONNECT AND SUMMARY
# -----------------------------------------

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

def get_db_summary():
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
# EMBEDDINGS AND METDATA    
# -----------------------------------------


def get_embedding(text: str):
    """Embed text using HuggingFace embeddings."""
    return embeddings.embed_query(text)

def match_embedding(text,k,col_name):
    q_emb = get_embedding(text)
    schema_hits = qdrant.search(collection_name=col_name,query_vector=q_emb,limit=k)
    retrieved = "\n".join([hit.payload["description"] for hit in schema_hits])
    return retrieved


def create_collections():
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

print("creating collections")

create_collections()

print("done!")


def create_embeddings(section_name):
    points = []
    for record in metadata[section_name]:
        text = str(record)
        emb = get_embedding(text)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload = {
            "section": section_name,       
            **record,                     
            "description": str(record)     
}
        ))
    qdrant.upsert(collection_name=section_name, points=points)

metadata = {}

# -----------------------------------------
# 1. TABLES
# -----------------------------------------


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



# -----------------------------------------
# 2. COLUMNS
# -----------------------------------------
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

# -----------------------------------------
# 3. PRIMARY KEYS
# -----------------------------------------
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

# -----------------------------------------
# 4. FOREIGN KEYS
# -----------------------------------------
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

# -----------------------------------------
# 5. INDEXES
# -----------------------------------------
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

# -----------------------------------------
# 6. CONSTRAINTS
# -----------------------------------------
metadata["constraints"] = query("""
    SELECT 
        TABLE_SCHEMA,
        TABLE_NAME,
        CONSTRAINT_NAME,
        CONSTRAINT_TYPE
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
    ORDER BY TABLE_SCHEMA, TABLE_NAME;
""")

# -----------------------------------------
# 7. CHECK CONSTRAINTS
# -----------------------------------------
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

# -----------------------------------------
# 8. TRIGGERS
# -----------------------------------------
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

# -----------------------------------------
# 9. VIEWS
# -----------------------------------------
metadata["views"] = query("""
    SELECT 
        TABLE_SCHEMA,
        TABLE_NAME,
        VIEW_DEFINITION
    FROM INFORMATION_SCHEMA.VIEWS;
""")

# -----------------------------------------
# 10. STORED PROCEDURES & FUNCTIONS
# -----------------------------------------
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


print("creating embeddings")
for section_name in metadata:
    create_embeddings(section_name)
print("done")


print("creating db summary")
db_summary = get_db_summary()
print("done")


# -----------------------------------------
# CrewAI: TOOLS
# -----------------------------------------

@tool
def context_extractor(prompt: str, collection_name: str,k: int):
    """Returns the most relevant context based on semantic similarity search."""
    context = match_embedding(prompt,k,collection_name)

    return context

@tool
def sql_tool(sql_query: str):
    """Executes or validates an SQL query against the database."""
    output = query(sql_query)

    if "SQL error Encountered" in output:
        data = {
           "success" : False,
           "error":output 
        }
        sql_error=None
        return data
    else:
        data = {
           "success" : True,
           "data":output 
        }
        sql_error=data
        return data





# -----------------------------------------
# CrewAI: AGENTS
# -----------------------------------------

coordinator_agent = Agent(
    name="Coordinator",
    backstory=(
        "You are a smart coordinator agent responsible for managing the workflow. "
        "You validate user queries, decide which tools to use, and route tasks "
        "to the appropriate agents. You must handle metadata extraction, data "
        "queries, and response generation intelligently."
    ),
    role="Decide workflow and route tasks",
    goal="""
    1. Validate user query.
    2. Use context_extractor to retrieve relevant context.
    3. If user requests data → route to SQL Agent.
    4. If user requests structure or explanation → route to Response Agent.
    """,
    tools=[context_extractor],
    llm=llm
)

sql_agent = Agent(
    name="SQL Agent",
    backstory=(
        "You are an expert SQL writer and debugger. Your job is to generate "
        "correct SQL queries based on user questions. You must call the SQL "
        "tool to validate queries, handle errors, retry failed queries up to "
        "5 times, and always return structured JSON with either data or a final error."
    ),
    role="SQL Writer and Debugger",
    llm=llm,
    tools=[sql_tool],
    goal="""
        Generate correct SQL queries based on user input.
        When you call the SQL tool, check the result:
        - If success == true → stop and return the data.
        - If success == false → fix the SQL and retry.

        You MUST follow this retry loop:
        1. Generate SQL
        2. Call sql_tool(sql=...)
        3. If error → fix according to the error and retry
        4. Maximum retries = 5
        5. If it still fails after 5 retries → output a JSON:
            {"final_error": "<error explanation>"}

        ALWAYS output JSON.
    """
)

response_agent = Agent(
    name="Response Generator",
    backstory=(
        "You are a natural-language response generator. Your goal is to create "
        "clear, helpful answers using all provided context, including metadata "
        "and SQL query results. You must handle missing data gracefully and "
        "produce human-readable final answers."
    ),
    role="Generate final natural-language responses",
    goal="Use provided context to answer user questions clearly.",
    tools=[],
    llm=llm
)



    

# -----------------------------------------
# CrewAI: TASKS
# -----------------------------------------






#----------------------
# Coordinator tasks
#----------------------


check_task = Task(
    description="""
        You are given two inputs:
        1. A database summary:
        {db_summary}

        2. A user prompt:
        {user_prompt}

        Your job is to return TWO decisions:

        ------------------------------------------
        1. is_true:
            • TRUE → The user is asking about the structure, schema, metadata,
              or anything answerable from the database.
            • FALSE → The question is irrelevant to the database.

        2. is_queryable:
            • TRUE → The user is asking about something that can be answered 
              by retrieving or inferring *actual data* (e.g. rows, values, counts).
            • FALSE → The user is only asking conceptual metadata (e.g. 
              "list tables", "what is the relation", "what columns exist").

        3. is_followUp:
            • TRUE → The user is asking a follow up question on the previous prompt and not asking an entirly new question.
            • FALSE → The user is following up on the previous prompt.      

        ------------------------------------------

        Output STRICTLY a valid JSON object of the form:

        {{
          "is_true": "TRUE" or "FALSE",
          "is_queryable": "TRUE" or "FALSE"
          "is_followUp": "TRUE" or "FALSE"
        }}
    """,
    agent=coordinator_agent,
    expected_output="A JSON object with fields: { 'is_true': 'TRUE' or 'FALSE', 'is_queryable': 'TRUE' or 'FALSE' }"
)


extractor_task = Task(
    description="""
        You are an intelligent database context extractor.
        Your mission: Analyze the user's question and strategically query the right metadata collections 
        to build a comprehensive context for answering their question.

        USER PROMPT:
        {user_prompt}

        AVAILABLE METADATA COLLECTIONS:
        - tables: Table metadata (schema, name, type, row counts, column names)
        - columns: Column details (data types, constraints, nullability, defaults)
        - primary_keys: Primary key definitions for all tables
        - foreign_keys: Foreign key relationships (parent/child table connections)
        - indexes: Index definitions (columns indexed, uniqueness, performance optimization)
        - constraints: All constraints (PK, FK, UNIQUE, CHECK)
        - check_constraints: Business rules and validation logic
        - triggers: Automated actions on data changes
        - views: Virtual table definitions
        - routines: Stored procedures and functions

        === YOUR EXTRACTION STRATEGY ===

        STEP 1: ANALYZE THE QUERY TYPE
        Identify what the user is asking about:
        - Schema exploration? → "what tables exist", "show me the database structure"
        - Relationships? → "how are X and Y connected", "what's the relationship between"
        - Data retrieval? → "get all", "show me", "find", "count", "sum"
        - Constraints? → "what are the rules", "validation", "requirements"
        - Performance? → "indexes", "optimization", "slow queries"

        STEP 2: EXTRACT KEY ENTITIES
        From the user prompt, identify:
        - Table names (even partial matches like "user" → "Users", "UserProfile")
        - Column names (like "price", "email", "date")
        - Concepts (like "sales", "inventory", "employee")
        - Actions (like "sold", "purchased", "hired")

        STEP 3: BUILD SEARCH PROMPTS
        Create semantic search prompts that capture the essence:
        - Use multiple related terms: "product item goods merchandise"
        - Include variations: "employee worker staff person"
        - Think domain-specific: "order purchase sale transaction"

        STEP 4: QUERY MULTIPLE COLLECTIONS
        Call context_extractor tool strategically with appropriate k values:
        - k=3-5 for specific entities (targeted search)
        - k=8-15 for exploratory queries (broad search)
        - k=10-20 for complex multi-table questions

        === DETAILED EXAMPLES ===

        Example 1: Relationship Query
        User: "What is the relationship between Orders and Customers?"
        Analysis: Need to understand table structure and how they connect
        Actions:
          → tool_call(prompt="orders customers", collection_name="tables", k=5)
          → tool_call(prompt="orders customers", collection_name="foreign_keys", k=8)
          → tool_call(prompt="orders customers", collection_name="primary_keys", k=6)
          → tool_call(prompt="orders customers relationship", collection_name="columns", k=10)

        Example 2: Data Retrieval Query
        User: "Show me the most sold products"
        Analysis: Need product tables, sales data, possibly order details
        Actions:
          → tool_call(prompt="product item goods", collection_name="tables", k=8)
          → tool_call(prompt="sales order quantity sold", collection_name="tables", k=8)
          → tool_call(prompt="product order sale", collection_name="foreign_keys", k=10)
          → tool_call(prompt="productID itemID", collection_name="columns", k=12)
          → tool_call(prompt="quantity amount sold price", collection_name="columns", k=10)

        Example 3: Schema Exploration
        User: "What tables are related to employees?"
        Analysis: Find all employee-related tables and their connections
        Actions:
          → tool_call(prompt="employee worker staff person hr", collection_name="tables", k=15)
          → tool_call(prompt="employee", collection_name="foreign_keys", k=12)
          → tool_call(prompt="employeeID staffID workerID", collection_name="columns", k=10)

        Example 4: Constraint Query
        User: "What are the validation rules for the User table?"
        Analysis: Need constraints, check constraints, and column rules
        Actions:
          → tool_call(prompt="user", collection_name="tables", k=3)
          → tool_call(prompt="user", collection_name="constraints", k=10)
          → tool_call(prompt="user", collection_name="check_constraints", k=8)
          → tool_call(prompt="user validation", collection_name="columns", k=12)

        Example 5: Multi-Table Complex Query
        User: "How do I track a customer's purchase history including shipping?"
        Analysis: Need customers, orders, order details, shipping/address tables
        Actions:
          → tool_call(prompt="customer client buyer", collection_name="tables", k=10)
          → tool_call(prompt="order purchase transaction", collection_name="tables", k=10)
          → tool_call(prompt="shipping delivery address", collection_name="tables", k=8)
          → tool_call(prompt="customer order shipping", collection_name="foreign_keys", k=15)
          → tool_call(prompt="customerID orderID", collection_name="primary_keys", k=8)

        Example 6: Performance Query
        User: "Which indexes exist on the Product table?"
        Analysis: Need table info and all related indexes
        Actions:
          → tool_call(prompt="product", collection_name="tables", k=3)
          → tool_call(prompt="product", collection_name="indexes", k=10)
          → tool_call(prompt="product", collection_name="columns", k=8)

        Example 7: Aggregation Query
        User: "Calculate total revenue by customer"
        Analysis: Need customer, order, and financial/amount columns
        Actions:
          → tool_call(prompt="customer client", collection_name="tables", k=6)
          → tool_call(prompt="order sale transaction", collection_name="tables", k=8)
          → tool_call(prompt="revenue price amount total cost", collection_name="columns", k=15)
          → tool_call(prompt="customer order", collection_name="foreign_keys", k=10)

        Example 8: Existence Check
        User: "Do we have supplier information?"
        Analysis: Search for supplier-related tables broadly
        Actions:
          → tool_call(prompt="supplier vendor provider manufacturer distributor", collection_name="tables", k=12)
          → tool_call(prompt="supplier vendor", collection_name="columns", k=10)

        === CRITICAL RULES ===

        1. ALWAYS call the tool multiple times - single calls rarely give complete context
        2. Be CREATIVE with search terms - think synonyms and related concepts
        3. Adjust k based on specificity:
           - Specific entity (known table name): k=3-5
           - General concept: k=10-15
           - Exploratory/complex: k=15-20
        4. For data queries, ALWAYS include:
           - Relevant table searches
           - Foreign key relationships
           - Column searches for data fields
        5. Think about the FULL data path:
           - What tables store the data?
           - How are they connected?
           - What columns contain the values?
        6. Output ALL retrieved context in your final answer

        Previous check result: {check_task.output}
    """,
    agent=coordinator_agent,
    context=[check_task],
    expected_output="A comprehensive JSON containing all queried collections and their retrieved metadata, organized by collection type.",
    condition=lambda context: check_task_should_run(context)
)

sql_error = None

sql_task = Task(
    description=f"""
        Given the user question and the database schema, 
        generate valid SQL using the retry loop defined in your agent.

        generate sql for sql server engine, use top instead of limit

        user question:
        {{user_prompt}}

        You will call sql_tool and recieve an output, if it an error call the tool again and try not to make the same mistake.

        error from previous call:
        {sql_error}
        
        Metadata context: {{extractor_task.output}}
    """,
    agent=sql_agent,
    context=[check_task, extractor_task],
    expected_output="Structured json with sql output or final error",
    condition=lambda context: sql_task_should_run(context)
)

response_task = Task(
    description="""
        Your job is to answer the user's question using all available output:

        The user question is:
        {user_prompt}

        Database summary:
        {db_summary}

        Use the outputs from previous tasks:
        - Check result: {check_task.output}
        - Extracted metadata context: {extractor_task.output}
        - SQL query result (if exists): {sql_task.output}

        Use whatever information exists.  
        If something is missing (null), ignore it.
    """,
    agent=response_agent,
    context=[check_task, extractor_task, sql_task],
    expected_output="Final response json"
)


def check_task_should_run(context):
    """Run extractor_task only if is_true is TRUE"""
    try:
        check_output = None

        if check_task in context:
            check_output = context[check_task]

        elif 'check_task' in context:
            check_output = context['check_task']

        elif hasattr(check_task, 'output') and check_task.output:
            check_output = check_task.output
        
        if check_output is None:
            print("[CONDITION CHECK] check_output is None, returning False")
            return False
        
        if hasattr(check_output, 'raw'):
            check_output = check_output.raw
            
        if isinstance(check_output, str):
            result = json.loads(check_output)
        else:
            result = check_output
            
        is_true = result.get('is_true', 'FALSE')
        print(f"[CONDITION CHECK] is_true = {is_true}")
        return is_true == 'TRUE'
    except Exception as e:
        print(f"[CONDITION ERROR] check_task_should_run: {e}")
        import traceback
        traceback.print_exc()
        return False




def sql_task_should_run(context):
    """Run sql_task only if is_queryable is TRUE"""
    try:
        check_output = None
        
        if check_task in context:
            check_output = context[check_task]

        elif 'check_task' in context:
            check_output = context['check_task']

        elif hasattr(check_task, 'output') and check_task.output:
            check_output = check_task.output
        
        if check_output is None:
            print("[CONDITION CHECK] check_output is None, returning False")
            return False
        
        if hasattr(check_output, 'raw'):
            check_output = check_output.raw
            
        if isinstance(check_output, str):
            result = json.loads(check_output)
        else:
            result = check_output
            
        is_queryable = result.get('is_queryable', 'FALSE')
        print(f"[CONDITION CHECK] is_queryable = {is_queryable}")
        return is_queryable == 'TRUE'
    except Exception as e:
        print(f"[CONDITION ERROR] sql_task_should_run: {e}")
        import traceback
        traceback.print_exc()
        return False


crew = Crew(
    agents=[
        coordinator_agent,
        sql_agent,
        response_agent
    ],
    tasks=[
        check_task,
        extractor_task,
        sql_task,
        response_task
    ],
    verbose=True
)


def chat_func(message, history):
    try:
        result = crew.kickoff(inputs={
            'user_prompt': message,
            'db_summary': db_summary
        })
        return result.raw
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


gr.ChatInterface(
    fn=chat_func, 
    type="messages"
).launch(share=True)

# what is the relation between purchaseorderdetail and purchase order header