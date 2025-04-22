import lancedb
from openai import OpenAI

from src.embeddings import Movie
from src.model_interaction import generate_with_openai


def build_hyde_query(client: OpenAI, query: str) -> str:
    prompt = f"""
Consider the following query, related to a movie: {query}
Create a one-sentence summary for a movie relevant to the query.
You do not need to describe an existing movie, feel free to invent, but stay relevant to the query.
    """
    return generate_with_openai(client, prompt)


def run_semantic_query(db: lancedb.DBConnection, query: str, limit: int = 3) -> list[Movie]:
    res: list[Movie] = db.open_table("movie").search(query).limit(limit).to_pydantic(Movie)
    return res


def format_movies(movies: list[Movie]) -> str:
    return "\n".join(
        [
            f"**{movie.title}** - Genre: {movie.genre} - Release: {movie.release_date} \n {movie.overview}"
            for movie in movies
        ]
    )


def run_reranking(
    client: OpenAI,
    movies: list[Movie],
    query: str,
) -> str:
    formated_movies = format_movies(movies)
    prompt = f"""
Consider the following query, related to a movie: {query}
The following movies were proposed as relevant to the query:
{formated_movies}
Please re-rank the movies based on their relevance to the query, removing any duplicate and any irrelevant item.
Only return the updated list, with all the information for each movie.
    """
    return generate_with_openai(client, prompt)


def answer_query_from_context(client: OpenAI, context: str, query: str) -> str:
    prompt = f"""
Consider the following query, related to a movie: {query}
The following movies were proposed as relevant to the query:

{context}

Provide an answer to the query, choosing the most relevant movie, in a friendly and open tone.
    """
    return generate_with_openai(client, prompt)
