from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
SOURCE_DB = "C:/PythonProject/cld/fdcpa.db"
LOCAL_DB = str(PROJECT_ROOT / "data" / "opinions.db")
FAISS_INDEX = str(PROJECT_ROOT / "data" / "faiss_index.bin")
FAISS_MAP = str(PROJECT_ROOT / "data" / "faiss_chunk_map.json")
CHECKPOINT_DIR = str(PROJECT_ROOT / "data" / "checkpoints")

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chunking
CHUNK_SIZE = 2000       # characters (~500 tokens)
CHUNK_OVERLAP = 400     # characters (~100 tokens)

# Search
DEFAULT_TOP_K = 10
OVERSAMPLE_FACTOR = 5   # fetch 5x results for grouping
