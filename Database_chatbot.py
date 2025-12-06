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
    temperature=0
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

schema_linking_agent = Agent(
    name="Schema Linking Agent",
    backstory=(
        "You are a database schema expert. You analyze user questions and identify "
        "exactly which tables, columns, foreign keys, and constraints are relevant. "
        "You output structured decisions, not raw data."
    ),
    role="Identify relevant schema elements",
    goal="Analyze the user question and output a structured list of schema elements to retrieve",
    tools=[],
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
        You are given four inputs:
        1. A database summary:
        {db_summary}

        2. A user prompt:
        {user_prompt}

        3. Chat history:
        {chat_history}

        4. Previously collected data and performed queries:
        {total_context}


        Your job is to return THREE decisions:

        ------------------------------------------
        1. is_true:
            • TRUE → The user is asking about the structure, schema, metadata,
              or anything answerable from the database.
            • FALSE → The question is irrelevant to the database.

        2. is_queryable:
            • TRUE → The user is asking about something that can be answered 
              by retrieving or inferring *actual data* (e.g. rows, values, counts) OR The user asks about a how a query should be written or even mentions a query.
            • FALSE → The user is only asking conceptual metadata (e.g. 
              "list tables", "what is the relation", "what columns exist") THAT IS NOT A QUERY.
        
        3. needs_rag:
            • TRUE → Need to extract new database context using RAG/embeddings because:
              - This is a NEW question (not a follow-up)
              - OR it's a follow-up but asks about DIFFERENT tables/entities than before
              - OR it's a follow-up that needs ADDITIONAL context not in total_context
              - The data in "Previously collected data and performed queries" is not sufficient enough to answer the user query
            • FALSE → Can answer using existing context because:
              - This is a simple follow-up asking for clarification/rephrasing
              - OR asking to modify/filter the previous query result
              - OR the "Previously collected data and performed queries" already contains all necessary schema information
              - OR simple conversational responses like "thank you"
            
            Examples:
            - "What tables exist?" → TRUE (new question, need context)
            - "Show me all customers" → TRUE (new question, need Customer table context)
            - "Now show me their orders" → TRUE (follow-up but NEW entity: orders)
            - "Sort that by date" → FALSE (follow-up on same data)
            - "What does that mean?" → FALSE (clarification request)
            - "Add a WHERE clause for status='active'" → FALSE (modifying existing query)
            - "How are Orders and Customers related?" → TRUE (need relationship context)
            - "Now also include Products" → TRUE (adding new entity, need more context)
            - "Explain the previous result" → FALSE (working with existing data)
            - "Can you make that query faster?" → FALSE (optimization of existing query)

        ------------------------------------------

        Output STRICTLY a valid JSON object of the form:

        {{
          "is_true": "TRUE" or "FALSE",
          "is_queryable": "TRUE" or "FALSE",
          "needs_rag": "TRUE" or "FALSE"
        }}
    """,
    agent=coordinator_agent,
    expected_output="A JSON object with fields: { 'is_true': 'TRUE' or 'FALSE', 'is_queryable': 'TRUE' or 'FALSE', 'needs_rag': 'TRUE' or 'FALSE' }"
)


schema_linking_task = Task(
    description="""
        You are a Schema Linking specialist.
        
        PREVIOUS CHECK RESULT:
        {check_task.output}
        
        === STEP 1: CHECK IF YOU SHOULD SKIP ===
        1. Parse the check_task.output JSON
        2. Look for the "needs_rag" field
        3. Look for the "is_true" field
        4. If EITHER needs_rag is "FALSE" OR is_true is "FALSE":
           - Output EXACTLY: {{"skip": true}}
           - STOP here, do NOT continue
        
        USER QUESTION:
        {user_prompt}
        
        DATABASE SUMMARY:
        {db_summary}
        
        PREVIOUS CONTEXT (if follow-up question):
        {total_context}
        
        === AVAILABLE METADATA COLLECTIONS ===
        
        **Core Structure Collections:**
        - **tables**: Metadata about each table (schema, name, type, engine, row stats)
        - **columns**: Column metadata (data types, length, nullability, defaults, extra details)
        - **primary_keys**: Primary key columns for every table
        - **foreign_keys**: Relationships between tables (parent/child columns)
        
        **Optimization & Performance:**
        - **indexes**: All indexes including type, uniqueness, and indexed columns
        
        **Business Rules & Validation:**
        - **constraints**: All constraints (PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK)
        - **check_constraints**: Check constraints and validation rules at table level
        
        **Advanced Database Objects:**
        - **triggers**: Trigger metadata and SQL definitions
        - **views**: View definitions and metadata for all SQL views
        - **routines**: Stored procedures and functions (definitions, return types)
        
        === WHEN TO USE EACH COLLECTION ===
        
        **Always Query:**
        - tables: For ANY query (foundation of schema understanding)
        - columns: For ANY query involving data retrieval or structure
        
        **Query for Relationships:**
        - foreign_keys: Multi-table queries, joins, "how are X and Y related"
        - primary_keys: Understanding table identity and relationships
        
        **Query for Performance Questions:**
        - indexes: "Which indexes exist?", "How is X indexed?", "Performance optimization"
        
        **Query for Business Rules:**
        - constraints: "What are the rules?", "What constraints exist?"
        - check_constraints: "Validation rules", "What values are allowed?"
        
        **Query for Advanced Features:**
        - triggers: "What triggers exist?", "Automated actions", "What happens when X changes?"
        - views: "What views exist?", "Show me virtual tables", "Summary tables"
        - routines: "What stored procedures?", "What functions?", "Database logic"
        
        === STEP 2: ANALYZE THE QUESTION ===
        
        Break down the user question to identify:
        - What tables/entities are mentioned or implied?
        - What type of data are they looking for?
        - Do they need to join multiple tables?
        - Are there any constraints or validation rules involved?
        - Are they asking about performance, triggers, views, or stored procedures?
        
        === STEP 3: CREATE YOUR RETRIEVAL PLAN ===
        
        Output a JSON with these fields:
        
        1. **table_keywords**: List of keyword phrases to search for tables
           - Combine related terms with spaces: "customer client buyer"
           - Think synonyms and variations
           - Example: ["customer client buyer", "order purchase sale", "product item goods"]
        
        2. **column_keywords**: List of keyword phrases to search for columns
           - Combine related column concepts with spaces
           - Think about what data fields would be needed
           - Example: ["customerID customer_id client_id", "price amount cost total", "quantity count number"]
        
        3. **needs_relationships**: true/false
           - true if question involves multiple tables or asks about connections
           - true if question asks "how", "relationship", "connected"
        
        4. **relationship_keywords**: List of keyword phrases for FK searches (only if needs_relationships is true)
           - Combine table names that need to be connected
           - Example: ["customer order", "order product", "customer address"]
        
        5. **needs_constraints**: true/false
           - true if asking about rules, validation, requirements, constraints
        
        6. **needs_indexes**: true/false
           - true if asking about performance, optimization, or indexes
        
        7. **needs_triggers**: true/false
           - true if asking about triggers, automated actions, or "what happens when"
        
        8. **needs_views**: true/false
           - true if asking about views, virtual tables, or summary tables
        
        9. **needs_routines**: true/false
           - true if asking about stored procedures, functions, or database logic
        
        10. **advanced_keywords**: List of keyword phrases for triggers/views/routines (only if any of needs_triggers/views/routines is true)
            - Example: ["customer order", "sales report", "calculate total"]
        
        11. **search_breadth**: Choose ONE:
            - "narrow": Simple single-table query → k_tables=15, k_columns=25, k_fk=30
            - "medium": 2-3 tables, moderate complexity → k_tables=35, k_columns=50, k_fk=60
            - "wide": Complex multi-table or exploratory → k_tables=60, k_columns=80, k_fk=100
        
        === EXAMPLES ===
        
        Example 1: Simple Query
        User: "Show me all customers"
        
        Analysis:
        - Single table (customers)
        - No joins needed
        - Need customer-related columns
        
        Output:
        {{
          "table_keywords": ["customer client buyer user"],
          "column_keywords": [
            "customerID customer_id client_id",
            "name first_name last_name full_name",
            "email email_address contact",
            "phone telephone mobile"
          ],
          "needs_relationships": false,
          "relationship_keywords": [],
          "needs_constraints": false,
          "needs_indexes": false,
          "needs_triggers": false,
          "needs_views": false,
          "needs_routines": false,
          "advanced_keywords": [],
          "search_breadth": "narrow"
        }}
        
        Example 2: Relationship Query
        User: "What is the relationship between Orders and Customers?"
        
        Analysis:
        - Two tables: orders, customers
        - Explicitly asking about relationships
        - Need foreign keys
        
        Output:
        {{
          "table_keywords": ["order orders purchase sale", "customer client buyer"],
          "column_keywords": [
            "orderID order_id purchase_id",
            "customerID customer_id client_id"
          ],
          "needs_relationships": true,
          "relationship_keywords": ["customer order", "order customer"],
          "needs_constraints": false,
          "needs_indexes": false,
          "needs_triggers": false,
          "needs_views": false,
          "needs_routines": false,
          "advanced_keywords": [],
          "search_breadth": "medium"
        }}
        
        Example 3: Complex Aggregation
        User: "Calculate total revenue by customer from completed orders"
        
        Analysis:
        - Multiple tables: customers, orders, possibly order_items
        - Need financial columns, status columns
        - Need relationships between tables
        - Aggregation means we need comprehensive context
        
        Output:
        {{
          "table_keywords": [
            "customer client buyer user",
            "order orders purchase sale transaction",
            "order_detail order_item line_item",
            "product item goods"
          ],
          "column_keywords": [
            "customerID customer_id client_id user_id",
            "orderID order_id purchase_id sale_id",
            "revenue total_amount price cost subtotal",
            "quantity amount count number",
            "status order_status state completed",
            "productID product_id item_id"
          ],
          "needs_relationships": true,
          "relationship_keywords": [
            "customer order",
            "order order_detail order_item",
            "order_detail product item"
          ],
          "needs_constraints": false,
          "needs_indexes": false,
          "needs_triggers": false,
          "needs_views": false,
          "needs_routines": false,
          "advanced_keywords": [],
          "search_breadth": "wide"
        }}
        
        Example 4: Constraint Query
        User: "What are the validation rules for the User table?"
        
        Analysis:
        - Single table focus (users)
        - Explicitly asking about validation/constraints
        - Need constraint information
        
        Output:
        {{
          "table_keywords": ["user users account member"],
          "column_keywords": [
            "userID user_id account_id",
            "username login user_name",
            "email email_address",
            "password pwd hash"
          ],
          "needs_relationships": false,
          "relationship_keywords": [],
          "needs_constraints": true,
          "needs_indexes": false,
          "needs_triggers": false,
          "needs_views": false,
          "needs_routines": false,
          "advanced_keywords": [],
          "search_breadth": "medium"
        }}
        
        Example 5: Performance Query
        User: "Which indexes exist on the Product table?"
        
        Analysis:
        - Asking about indexes specifically
        - Need product table info and indexes
        
        Output:
        {{
          "table_keywords": ["product item goods merchandise"],
          "column_keywords": ["productID product_id item_id"],
          "needs_relationships": false,
          "relationship_keywords": [],
          "needs_constraints": false,
          "needs_indexes": true,
          "needs_triggers": false,
          "needs_views": false,
          "needs_routines": false,
          "advanced_keywords": ["product item"],
          "search_breadth": "narrow"
        }}
        
        Example 6: Triggers Query
        User: "What happens when an order is created?"
        
        Analysis:
        - Asking about automated actions
        - Need triggers related to orders
        - Might also need views or routines
        
        Output:
        {{
          "table_keywords": ["order orders purchase sale"],
          "column_keywords": ["orderID order_id status"],
          "needs_relationships": false,
          "relationship_keywords": [],
          "needs_constraints": false,
          "needs_indexes": false,
          "needs_triggers": true,
          "needs_views": false,
          "needs_routines": false,
          "advanced_keywords": ["order create insert new"],
          "search_breadth": "medium"
        }}
        
        Example 7: Views Query
        User: "Show me all sales summary views"
        
        Analysis:
        - Explicitly asking about views
        - Sales-related virtual tables
        
        Output:
        {{
          "table_keywords": ["sales revenue order transaction"],
          "column_keywords": ["sales revenue total amount"],
          "needs_relationships": false,
          "relationship_keywords": [],
          "needs_constraints": false,
          "needs_indexes": false,
          "needs_triggers": false,
          "needs_views": true,
          "needs_routines": false,
          "advanced_keywords": ["sales summary report total"],
          "search_breadth": "medium"
        }}
        
        Example 8: Stored Procedures Query
        User: "What stored procedures calculate customer loyalty points?"
        
        Analysis:
        - Asking about stored procedures/functions
        - Customer and loyalty point calculation logic
        
        Output:
        {{
          "table_keywords": ["customer client loyalty points rewards"],
          "column_keywords": ["customerID points loyalty_points"],
          "needs_relationships": false,
          "relationship_keywords": [],
          "needs_constraints": false,
          "needs_indexes": false,
          "needs_triggers": false,
          "needs_views": false,
          "needs_routines": true,
          "advanced_keywords": ["customer loyalty points calculate reward"],
          "search_breadth": "medium"
        }}
        
        Example 9: Comprehensive Schema Exploration
        User: "Tell me everything about the Order processing system"
        
        Analysis:
        - Very exploratory
        - Need tables, relationships, triggers, views, procedures
        - Comprehensive retrieval
        
        Output:
        {{
          "table_keywords": [
            "order orders purchase sale transaction",
            "order_detail order_item line_item",
            "customer client buyer",
            "product item goods",
            "payment transaction",
            "shipping delivery"
          ],
          "column_keywords": [
            "orderID order_id purchase_id",
            "customerID customer_id",
            "productID product_id",
            "status order_status state",
            "total amount price"
          ],
          "needs_relationships": true,
          "relationship_keywords": [
            "order customer",
            "order order_detail",
            "order_detail product",
            "order payment",
            "order shipping"
          ],
          "needs_constraints": true,
          "needs_indexes": true,
          "needs_triggers": true,
          "needs_views": true,
          "needs_routines": true,
          "advanced_keywords": ["order process payment shipping calculate"],
          "search_breadth": "wide"
        }}
        
        === CRITICAL GUIDELINES ===
        
        1. **Combine keywords with spaces**: 
           - GOOD: "customer client buyer user"
           - BAD: ["customer", "client", "buyer", "user"]
        
        2. **Think about variations**:
           - Different naming conventions: customer_id, customerID, CustomerId
           - Synonyms: client, buyer, user, account
           - Abbreviations: dept vs department, qty vs quantity
        
        3. **Be comprehensive for column_keywords**:
           - Include ID fields
           - Include data fields mentioned in question
           - Include related fields that might be needed
           - Group semantically related terms together
        
        4. **Set the right boolean flags**:
           - needs_relationships: Multi-table queries, joins, "how are they connected"
           - needs_constraints: "rules", "validation", "requirements", "constraints"
           - needs_indexes: "performance", "optimization", "indexes", "slow queries"
           - needs_triggers: "what happens when", "automated", "on insert/update/delete"
           - needs_views: "views", "virtual tables", "summary tables", "reports"
           - needs_routines: "stored procedure", "function", "logic", "calculate"
        
        5. **Use advanced_keywords when appropriate**:
           - Only populate if needs_indexes, needs_triggers, needs_views, or needs_routines is true
           - Combine relevant terms for searching these advanced objects
        
        6. **Choose search_breadth wisely**:
           - narrow: You know exactly what table(s) you need, simple query
           - medium: 2-3 tables, standard relationships, or asking about specific advanced objects
           - wide: Complex query, multiple joins, exploratory, or comprehensive schema analysis
        
        7. **Output ONLY valid JSON, nothing else**
    """,
    agent=schema_linking_agent,
    context=[check_task],
    expected_output="A JSON object specifying keyword-based retrieval plan with combined search terms and collection flags"
)

# -----------------------------------------
# PYTHON FUNCTION: Execute Retrieval Plan
# -----------------------------------------

def execute_retrieval_plan(schema_plan_output):
    """
    Parses the schema linking plan JSON and executes all RAG calls in Python.
    Returns aggregated metadata results.
    """
    try:
        # Parse the schema linking plan
        plan = json.loads(schema_plan_output)
        
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
        
        # Determine k values based on search_breadth
        k_values = {
            "narrow": {
                "tables": 15, "columns": 25, "foreign_keys": 30,
                "primary_keys": 20, "constraints": 25, "indexes": 20,
                "triggers": 15, "views": 15, "routines": 15
            },
            "medium": {
                "tables": 35, "columns": 50, "foreign_keys": 60,
                "primary_keys": 40, "constraints": 45, "indexes": 40,
                "triggers": 30, "views": 30, "routines": 30
            },
            "wide": {
                "tables": 60, "columns": 80, "foreign_keys": 100,
                "primary_keys": 70, "constraints": 70, "indexes": 60,
                "triggers": 50, "views": 50, "routines": 50
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

extraction_execution_task = Task(
    description="""
        RETRIEVED METADATA CONTEXT:
        {extracted_metadata}
        
        Simply output the extracted metadata context that was provided.
    """,
    agent=coordinator_agent,
    context=[check_task, schema_linking_task],
    expected_output="Retrieved metadata context from Python-based RAG execution"
)

query_plan_task = Task(
    description="""
        You are a SQL Query Planner using Chain-of-Thought reasoning.
        
        PREVIOUS CHECK RESULT:
        {check_task.output}
        
        === FIRST: CHECK IF YOU SHOULD SKIP ===
        Parse the check_task.output JSON.
        If is_queryable is "FALSE", output {{"skip": true, "reason": "No query needed"}} and end immediately.
        
        USER QUESTION:
        {user_prompt}
        
        EXTRACTED SCHEMA CONTEXT:
        {extraction_execution_task.output}
        
        PREVIOUS CONTEXT:
        {total_context}
        
        === YOUR TASK: CREATE A STEP-BY-STEP QUERY PLAN ===
        
        Using Chain-of-Thought reasoning, generate a detailed execution plan that explains:
        
        **Step 1: Identify Main Entity**
        - What is the primary table we need to query?
        - What is the main information the user is asking for?
        
        **Step 2: Identify Required Joins**
        - What other tables do we need?
        - How are they connected? (foreign key relationships)
        - What is the join path from the main table to related tables?
        
        **Step 3: Determine Filter Conditions**
        - What WHERE conditions are needed?
        - Are there any specific values to filter by?
        - Any date ranges or status conditions?
        
        **Step 4: Plan Aggregations (if needed)**
        - Do we need COUNT, SUM, AVG, MIN, MAX?
        - What columns do we GROUP BY?
        - Any HAVING conditions?
        
        **Step 5: Determine Sorting and Limiting**
        - Should results be ordered? By which column(s)?
        - Is there a limit on number of rows (TOP N)?
        
        **Step 6: Handle Special Cases**
        - Any DISTINCT needed to remove duplicates?
        - Any subqueries required?
        - Any UNION/INTERSECT/EXCEPT operations?
        
        === CRITICAL RULES ===
        
        1. **DO NOT generate SQL code** - only the reasoning plan
        2. Use the extracted schema context to identify exact table and column names
        3. Explain WHY each step is needed based on the user's question
        4. If information is missing, state what assumptions you're making
        5. Output your plan as structured text, not code
        
        === EXAMPLE OUTPUT FORMAT ===
        
        Query Plan:
        
        Step 1: Main Entity
        - Primary table: Customer
        - Goal: Find customer purchase totals
        
        Step 2: Joins Needed
        - Join Customer to Orders via Customer.CustomerID = Orders.CustomerID
        - Join Orders to OrderDetails via Orders.OrderID = OrderDetails.OrderID
        
        Step 3: Filters
        - WHERE Orders.OrderDate >= '2024-01-01'
        - Only include completed orders (Status = 'Completed')
        
        Step 4: Aggregations
        - SUM(OrderDetails.Quantity * OrderDetails.UnitPrice) as TotalPurchase
        - GROUP BY Customer.CustomerID, Customer.Name
        
        Step 5: Sorting
        - ORDER BY TotalPurchase DESC
        - TOP 10 customers
        
        Step 6: Special Cases
        - None needed for this query
    """,
    agent=sql_agent,
    context=[check_task, schema_linking_task, extraction_execution_task],
    expected_output="A detailed Chain-of-Thought query execution plan explaining each step (NOT SQL code)"
)

sql_task = Task(
    description="""
        PREVIOUS CHECK RESULT:
        {check_task.output}
        
        === FIRST: CHECK IF YOU SHOULD SKIP ===
        Parse the check_task.output JSON.
        If is_queryable is "FALSE", output {{"skip": true, "reason": "No query needed"}} and end immediately.
        
        Also check query_plan_task.output - if it contains {{"skip": true}}, output {{"skip": true}} and end.

        Now you have a step-by-step query plan from the Query Plan Agent.
        Your job is to convert this plan into actual SQL code.

        QUERY PLAN:
        {query_plan_task.output}

        USER QUESTION:
        {user_prompt}
        
        METADATA CONTEXT:
        {extraction_execution_task.output}

        PREVIOUS QUERIES:
        {total_context}

        === YOUR TASK: GENERATE SQL ===
        
        Follow the query plan step-by-step to generate valid SQL Server syntax.
        - Use TOP instead of LIMIT
        - Use correct table and column names from metadata
        - Follow SQL Server conventions
        
        === ERROR HANDLING LOOP ===
        1. Generate SQL query
        2. Call sql_tool(sql_query="your SQL here")
        3. Check the result:
           - If success == true: Output {{"success": true, "sql_query": "...", "data": [...]}}
           - If success == false: Read the error, fix the SQL, and retry
        4. Maximum 5 retry attempts
        5. After 5 failures, output {{"final_error": "explanation of what went wrong"}}

        ALWAYS output valid JSON with either:
        - {{"success": true, "sql_query": "...", "data": [...]}}
        - {{"final_error": "..."}} (after 5 failed attempts)
        - {{"skip": true}} (if skipping)

    """,
    agent=sql_agent,
    context=[check_task, schema_linking_task, extraction_execution_task, query_plan_task],
    expected_output="Structured json with sql query and output or final error",
)

response_task = Task(
    description="""
        Your job is to answer the user's question using all available output:

        DATABASE SUMMARY:
        {db_summary}

        USER QUESTION:
        {user_prompt}

        CHAT HISTORY:
        {chat_history}

        === OUTPUTS FROM PREVIOUS TASKS ===
        
        Check result: {check_task.output}
        Schema linking plan: {schema_linking_task.output}
        Extracted metadata context: {extraction_execution_task.output}
        Query execution plan: {query_plan_task.output}
        SQL query result: {sql_task.output}
        Previously collected data: {total_context}

        === YOUR TASK: GENERATE FINAL RESPONSE ===
        
        1. Parse all the outputs above (they may contain JSON)
        2. If any task was skipped (contains {{"skip": true}}), understand why
        3. Use whatever information is available to answer the user
        
        **Response Guidelines:**
        - If the user asked for data and SQL was executed successfully:
          * State the SQL query used
          * Show up to 10 sample rows (unless user asks for more)
          * Summarize the results in natural language
        
        - If the user asked about schema/structure (no query needed):
          * Use the extracted metadata context
          * Explain relationships, tables, columns clearly
        
        - If there was an error:
          * Explain what went wrong
          * Suggest how the user can rephrase or fix their question
        
        - If the question is irrelevant to the database:
          * Politely explain you can only answer database-related questions
          * Suggest what types of questions you can help with
        
        **Quality Rules:**
        - Be concise but complete
        - Use bullet points for lists
        - Format data in readable tables when appropriate
        - If information is missing or null, skip it gracefully
        - Encourage specificity if the question is too broad
    """,
    agent=response_agent,
    context=[check_task, schema_linking_task, extraction_execution_task, query_plan_task, sql_task],
    expected_output="Final response json"
)




crew = Crew(
    agents=[
        coordinator_agent,
        schema_linking_agent,
        sql_agent,
        response_agent
    ],
    tasks=[
        check_task,
        schema_linking_task,
        extraction_execution_task,
        query_plan_task,
        sql_task,
        response_task
    ],
    verbose=True,
    tracing=True
)

collected_context = []

def chat_func(message, history):
    try:
        # First, do a preliminary run to get schema linking output
        # We need to execute schema_linking_task first to get the plan
        preliminary_crew = Crew(
            agents=[coordinator_agent, schema_linking_agent],
            tasks=[check_task, schema_linking_task],
            verbose=False
        )
        
        preliminary_result = preliminary_crew.kickoff(inputs={
            'user_prompt': message,
            'db_summary': db_summary,
            'chat_history': history,
            'total_context': collected_context
        })
        
        # Get schema linking output
        schema_plan = schema_linking_task.output.raw if schema_linking_task.output else "{}"
        
        # Execute retrieval in Python
        extraction_output = execute_retrieval_plan(schema_plan)
        
        # Now run the full crew with extraction results injected
        result = crew.kickoff(inputs={
            'user_prompt': message,
            'db_summary': db_summary,
            'chat_history': history,
            'total_context': collected_context,
            'extracted_metadata': extraction_output  # Inject the Python-generated extraction results
        })
        try:
            # Use the Python-generated extraction output directly
            extraction_data = str(extraction_output)
            
            # Get SQL output and parse JSON
            sql_output = sql_task.output.raw if sql_task.output else "{}"
            sql_data = json.loads(sql_output) if sql_output != "None" else {}
            sql_query = sql_data.get("sql_query", "None")
            
            collected_context.append(f"""extracted data: {extraction_data}
                                        performed queries: {sql_query}
                                    """)
        except Exception as e:
            print(f"Error collecting context: {e}")
            collected_context.append("Error collecting context")
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