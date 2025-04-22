from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_PATH = DATA_DIR / 'movies.parquet'
LANCEDB_URI = DATA_DIR / 'lancedb'
