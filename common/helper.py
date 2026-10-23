from datasets import load_dataset
import numpy as np
import os
import psycopg2
import clickhouse_connect
import uuid
from qdrant_client import QdrantClient
from common.const import (
    DATASET, LOCAL_CATEGORIES_FILE_PATH,
    SUPABASE_DATABASE, SUPABASE_USERNAME, SUPABASE_PASSWORD, SUPABASE_HOST, SUPABASE_PORT, # Supabase
    CLICKHOUSE_HOST, CLICKHOUSE_USERNAME, CLICKHOUSE_PASSWORD, CLICKHOUSE_PORT , # ClickHouse
    QDRANT_CLUSTER_ENDPOINT, MONGODB_URI, MONGODB_DATABASE , CHROMADB_PATH
)
from pymongo import MongoClient
import chromadb


# =====================================
# Load data from HuggingFace
# =====================================

def load_hf_data(root_split, data_type, category, split=None):
    """
    Fetches the path for specific dataset configurations.

    Args:
        root_split (str): The root split directory type. Options: 'raw', '0core', '5core'.
        data_type (str): The specific feature set to load.
        split: Data split. Required for core roots.
        
    Valid Configurations:
        - root_split = "raw" → data_type ∈ {"meta", "review"} → split = "full"
        - root_split ∈ {"0core", "5core"} → data_type = "rating_only" → split = "full"
        - root_split ∈ {"0core", "5core"} → data_type ∈ {"last_out", "last_out_w_his", "timestamp", "timestamp_w_his"} → split ∈ {"train", "val", "test"}
    """
    
    if (root_split == "raw") | ("data_type" == "rating_only"):
        split = "full"
    
    # Load data
    data = load_dataset(
        DATASET,
        f"{root_split}_{data_type}_{category}",
        split=split,
        streaming=True,
        trust_remote_code=True,
        features = None
    )
    return data
        
def get_categories(file_path=LOCAL_CATEGORIES_FILE_PATH):
    with open(file_path, "r") as f:
        categories = [line.strip() for line in f]
    return categories
        

# =====================================
# PostgreSQL
# =====================================

def connect_postgresql(dbname=SUPABASE_DATABASE, user=SUPABASE_USERNAME, password=SUPABASE_PASSWORD, host=SUPABASE_HOST, port=SUPABASE_PORT):
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn

def measure_postgres_query_time(conn, query, params=None):
    explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON) {query}"

    with conn.cursor() as cur:
        cur.execute(explain_query, params)
        result = cur.fetchone()[0]  # JSON result

    execution_time = result[0]["Execution Time"]  # in ms
    planning_time = result[0]["Planning Time"]

    return execution_time + planning_time 

# =====================================
# ClickHouse
# =====================================

def connect_clickhouse(host=CLICKHOUSE_HOST, user=CLICKHOUSE_USERNAME, password=CLICKHOUSE_PASSWORD, port=CLICKHOUSE_PORT):
    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        secure=True
    )
    return client

def clickhouse_extract_data_as_df(query, verbose=False):
    client = connect_clickhouse()
    if verbose:
        print("connected to clickhouse")
    df = client.query_df(query)
    client.close()
    
    return df

def clickhouse_extract_data_as_list(query):
    client = connect_clickhouse()
    out = client.query(query).result_set
    client.close()
    
    return out

def measure_clickhouse_query_performance(user_query):
    """
    Measure ClickHouse query performance using system.query_log
    """
    user_query = user_query.strip().rstrip(";")
    client = connect_clickhouse()

    # Generate a unique query_id
    qid = str(uuid.uuid4())

    # Run the query with a known query_id
    client.query(user_query, settings={"query_id": qid})

    # Flush logs so the query appears in query_log
    client.command("SYSTEM FLUSH LOGS")

    # Retrieve execution metrics for THIS query
    performance_query = """
    SELECT
        query_id,
        query_duration_ms,
        read_rows,
        read_bytes,
        query
    FROM system.query_log
    WHERE
        type = 'QueryFinish'
        AND query_id = %(qid)s
    LIMIT 1
    """

    performance_df = client.query_df(performance_query,parameters={"qid": qid})
    query_duration_ms = performance_df['query_duration_ms'][0]
    
    client.close()
    
    return query_duration_ms, performance_df

# =====================================
# Vector Database
# =====================================

# Qdrant
def connect_qdrant(url=QDRANT_CLUSTER_ENDPOINT, api_key=os.environ["QDRANT_API_KEY"]):
    qd_client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=1200
    )
    return qd_client




# =====================================
# Data Processing
# =====================================

### Products

def parse_price(value):
    """
    Correct the values of "price" from data:
    - If value = "None", convert to python None
    - If value is a number, convert to number (instead of text)
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip()

        if value.lower() in {"none", "", "null"}:
            return None

        # Optional: remove currency symbols
        value = value.replace("$", "").replace(",", "")

        try:
            return float(value)
        except ValueError:
            return None

    return None

def parse_product_images(images):
    first_image = None
    for hi_res_image in images['hi_res']:
        if hi_res_image is not None:
            first_image = hi_res_image # Take only the first high resolution image. The idea is to vectorize the image for the subsequent tasks.
            break
    
    image_count = len([i for i in images['hi_res'] if i is not None])
    variant_count = len(images['variant'])
    
    return first_image, image_count, variant_count

def convert_list_of_values_to_string(list_of_values):
    """Convert value to string, handling lists/arrays"""
    if (list_of_values is None) | (len(list_of_values) == 0):
        return None
    elif isinstance(list_of_values, list):
        return ' '.join(val.strip() for val in list_of_values)
    else:
        return list_of_values.strip()
    

### Reviews
def parse_review_images(images):
    image_count = len(images)
    
    if image_count > 0:
        first_image = images[0]['large_image_url']
    else:
        first_image = None
    
    return first_image, image_count

# =====================================
# MongoDB
# =====================================

def connect_mongodb(uri=MONGODB_URI, database=MONGODB_DATABASE):
    """Connect to MongoDB Atlas and return database object"""
    client = MongoClient(uri)
    db = client[database]
    return db, client


def get_mongodb_collection_stats(db, collection_name):
    """Get statistics about a MongoDB collection"""
    collection = db[collection_name]
    count = collection.count_documents({})
    sample = collection.find_one()
    return {
        'count': count,
        'sample': sample
    }


# =====================================
# ChromaDB
# =====================================

def connect_chromadb(path=CHROMADB_PATH):
    """Connect to ChromaDB and return client"""
    client = chromadb.PersistentClient(path=path)
    return client

def get_or_create_chromadb_collection(client, collection_name):
    """Get existing collection or create new one"""
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' found")
    except:
        collection = client.create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created")
    return collection

def get_chromadb_collection_stats(collection):
    """Get statistics about a ChromaDB collection"""
    count = collection.count()
    return {'count': count}


# =====================================
# Similarity
# =====================================

def dot_product(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return dot_product(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))