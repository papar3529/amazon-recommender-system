import os
from dotenv import load_dotenv
import torch
load_dotenv() #this loads the .env file

# ============================================
# SET UP
# ============================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET = "McAuley-Lab/Amazon-Reviews-2023"
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)
SPLIT_LIST = ["train", "valid", "test"]

END_TIMESTAMP_5CORE_TRAIN = 1628643414042
START_TIMESTAMP_5CORE_TRAIN = 1470877014042 # 5 years ago from END_TIMESTAMP_5CORE_TRAIN - The history of actual data is more than this but we only use it for recency. Choosing this number so the train data is substantially large compared to validation and testing set

# Categories
HF_CATEGORIES_FILE = "all_categories.txt"
LOCAL_CATEGORIES_FILE_PATH = os.path.join(PROJECT_ROOT, "data", HF_CATEGORIES_FILE)


# ============================================
# GENERAL DATABASE SETUP
# ============================================

# Database and tables
BATCH_SIZE = 5000
SCHEMA = "amazon"
TABLE_RAW_REVIEWS = "raw_reviews"
TABLE_RAW_PRODUCTS = "raw_products"
TABLE_FEATURED_DENSE_VECTORS = "featured_dense_vectors"
TABLE_FEATURED_USER_VECTORS = "featured_user_vectors"
TABLE_USER_AGGREGATED_STATE = "user_aggregated_state"
TABLE_MATERIALIZED_VIEW_USER = "user_materialized_view"
TABLE_USER_ITEM_STATE = "user_item_state"
TABLE_USER_ITEM_STATE_SNAPSHOT = "user_item_snapshot"
TABLE_VECTOR_PRODUCTS = "vector_products"
TABLE_RECENT_ITEM_REVIEW_SUMMARY = "recent_item_review_summary" 

PARQUET_FOLDER_RAW_PRODUCTS = os.path.join("/home", "vuongtc", "Documents", f"{TABLE_RAW_PRODUCTS}_parquet")
PARQUET_FOLDER_RAW_REVIEWS = os.path.join("/home", "vuongtc", "Documents", f"{TABLE_RAW_REVIEWS}_parquet")

TABLE_5CORE_TIMESTAMP_W_HIS_DICT = {}
for split in SPLIT_LIST:
    TABLE_5CORE_TIMESTAMP_W_HIS_DICT[split] = f"5core_timestamp_w_his_{split}"

TABLE_5CORE_TIMESTAMP_W_HIS_TRAIN_USER_ITEM_INTERACTIONS = "5core_timestamp_w_his_train_user_item_interactions"

PARQUET_FOLDERS_5CORE_TIMESTAMP_W_HIS = {}
for split in SPLIT_LIST:
    PARQUET_FOLDERS_5CORE_TIMESTAMP_W_HIS[split] = os.path.join("/home", "vuongtc", "Documents", f"5core_timestamp_w_his_{split}_parquet")

PRODUCT_VECTOR_DB_COLLECTION = "product_embeddings"
REVIEW_VECTOR_DB_COLLECTION = "review_embeddings"

# MongoDB
MONGODB_CONNECTION_STRING = "mongodb://localhost:27017"
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = "amazon_reviews"
MONGODB_PRODUCTS_COLLECTION = "products"
MONGODB_REVIEWS_COLLECTION = "reviews"
MONGODB_BATCH_SIZE = 1000

# PostgreSQL 
POSTGRESQL_DATABASE = "postgres"
POSTGRESQL_HOST = "localhost"
POSTGRESQL_PORT = 5432
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Supabase
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_DATABASE = "postgres"
SUPABASE_USERNAME = os.getenv("SUPABASE_USERNAME")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
SUPABASE_PORT = 5432

# ClickHouse
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_USERNAME = os.getenv("CLICKHOUSE_USERNAME")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_PORT = 443


# ============================================
# VECTOR DATABASE
# ============================================

VECTOR_DB_TABLE_REVIEW_EMBEDDINGS = "review_embeddings"
VECTOR_DB_TABLE_PRODUCT_EMBEDDINGS = "product_embeddings"

# Qdrant - AWS in London
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_CLUSTER_ENDPOINT_MINH_HOANG = os.getenv("QDRANT_CLUSTER_ENDPOINT_MINH_HOANG")

# ============================================
# ELASTICSEARCH SETUP (London Cloud)
# ============================================
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
#===============================================

# ============================================
# CHROMADB SETUP
# ============================================
CHROMADB_PATH = os.path.join(PROJECT_ROOT, "data", "chromadb")
CHROMADB_COLLECTION_PRODUCTS = "product_embeddings"
CHROMADB_COLLECTION_REVIEWS = "review_embeddings"
