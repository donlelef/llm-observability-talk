from datetime import datetime

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

EMBEDDING_FUNCTION = get_registry().get("openai").create(name="text-embedding-3-small")


class Movie(LanceModel):
    id: int
    title: str
    release_date: datetime
    runtime: int
    genre: str
    overview: str = EMBEDDING_FUNCTION.SourceField()
    vector: Vector(EMBEDDING_FUNCTION.ndims()) = EMBEDDING_FUNCTION.VectorField()
