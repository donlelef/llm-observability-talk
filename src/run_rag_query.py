import logging
import uuid

import lancedb
from langfuse.decorators import observe, langfuse_context
from langfuse.openai import OpenAI

from src.embeddings import Movie
from src.paths import LANCEDB_URI


@observe
def generate_with_openai(client: OpenAI, prompt: str) -> str:
    langfuse_context.update_current_observation(input={"prompt": prompt})
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": prompt},
        ],
        seed=42,
        temperature=0,
    )
    return response.choices[0].message.content


@observe
def build_hyde_query(client: OpenAI, query: str) -> str:
    langfuse_context.update_current_observation(input={"query": query})
    prompt = f"""
Consider the following query, related to a movie: {query}
Create a one-sentence summary for a movie relevant to the query.
You do not need to describe an existing movie, feel free to invent, but stay relevant to the query.
    """
    return generate_with_openai(client, prompt)


def run_semantic_query(
    db: lancedb.DBConnection, query: str, limit: int = 3
) -> list[Movie]:
    res: list[Movie] = (
        db.open_table("movie").search(query).limit(limit).to_pydantic(Movie)
    )
    return res


def format_movies(movies: list[Movie]) -> str:
    return "\n".join(
        [
            f"**{movie.title}** - Genre: {movie.genre} - Release: {movie.release_date} \n {movie.overview}"
            for movie in movies
        ]
    )


@observe
def run_reranking(
    client: OpenAI,
    movies: list[Movie],
    query: str,
) -> str:
    langfuse_context.update_current_observation(
        input={"query": query, "movies": movies}
    )
    formated_movies = format_movies(movies)
    prompt = f"""
Consider the following query, related to a movie: {query}
The following movies were proposed as relevant to the query:
{formated_movies}
Please re-rank the movies based on their relevance to the query, removing any duplicate and any irrelevant item.
Only return the updated list, with all the information for each movie.
    """
    return generate_with_openai(client, prompt)


@observe
def answer_query_from_context(client: OpenAI, context: str, query: str) -> str:
    langfuse_context.update_current_observation(
        input={"query": query, "context": context}
    )
    prompt = f"""
Consider the following query, related to a movie: {query}
The following movies were proposed as relevant to the query:

{context}

Provide an answer to the query, choosing the most relevant movie, in a friendly and open tone.
    """
    return generate_with_openai(client, prompt)


@observe
def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    client = OpenAI()
    query = "I would like to watch a movie with dragons"
    session_id = uuid.uuid4().hex
    langfuse_context.update_current_trace(session_id=session_id)
    logging.info(f"Session ID: {session_id}")

    logging.info(f"Query: {query}")
    logging.info("Building hyde query...")
    hyde_query = build_hyde_query(client, query)
    logging.info(f"Hyde query: {hyde_query}...")

    logging.info("Connecting to LanceDB...")
    db = lancedb.connect(LANCEDB_URI)
    logging.info("Running queries...")
    res: list[Movie] = run_semantic_query(db, query)
    hyde_res: list[Movie] = run_semantic_query(db, hyde_query)
    logging.info(
        f"Length for original query {len(res)}, length of hyde query: {len(hyde_res)}"
    )

    logging.info("Re-ranking results...")
    reranked_movies = run_reranking(client, res + hyde_res, query)
    logging.info(f"Reranked movies: {reranked_movies}")

    logging.info("Answering query...")
    answer = answer_query_from_context(client, reranked_movies, query)
    logging.info(f"Answer: {answer}")


if __name__ == "__main__":
    main()
