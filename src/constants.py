"""
Constants for the project.
Support for environment variables.
"""
import os
from typing import Dict, Any

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_URI = os.getenv("MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")


ETCD_HOST = "127.0.0.1"
ETCD_PORT = "2379"
ETCD_PREFIX = "/llama_learning/config"



