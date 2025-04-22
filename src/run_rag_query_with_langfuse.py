import logging
import uuid

import lancedb
from langfuse.decorators import observe, langfuse_context
from langfuse.openai import OpenAI

from src.embeddings import Movie
from src.paths import LANCEDB_URI
from src.rag import build_hyde_query, run_reranking, answer_query_from_context, run_semantic_query
from src.scores import ThumbScore


@observe
def observed_build_hyde_query(client: OpenAI, query: str) -> str:
    langfuse_context.update_current_observation(input={"query": query})
    return build_hyde_query(client, query)


@observe
def observed_run_reranking(
    client: OpenAI,
    movies: list[Movie],
    query: str,
) -> str:
    langfuse_context.update_current_observation(input={"query": query, "movies": movies})
    return run_reranking(client, movies, query)


@observe
def observed_answer_query_from_context(client: OpenAI, context: str, query: str) -> str:
    langfuse_context.update_current_observation(input={"query": query, "context": context})
    return answer_query_from_context(client, context, query)


@observe
def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    client = OpenAI()
    query = "I would like to watch a movie with dragons"
    session_id = uuid.uuid4().hex
    user_id = "AnonimizedLele"
    langfuse_context.update_current_trace(session_id=session_id, user_id=user_id)
    logging.info(f"Session ID: {session_id}")

    logging.info(f"Query: {query}")
    logging.info("Building hyde query...")
    hyde_query = observed_build_hyde_query(client, query)
    logging.info(f"Hyde query: {hyde_query}...")

    logging.info("Connecting to LanceDB...")
    db = lancedb.connect(LANCEDB_URI)
    logging.info("Running queries...")
    res: list[Movie] = run_semantic_query(db, query)
    hyde_res: list[Movie] = run_semantic_query(db, hyde_query)
    logging.info(f"Length for original query {len(res)}, length of hyde query: {len(hyde_res)}")

    logging.info("Re-ranking results...")
    reranked_movies = observed_run_reranking(client, res + hyde_res, query)
    logging.info(f"Reranked movies: {reranked_movies}")

    logging.info("Answering query...")
    answer = observed_answer_query_from_context(client, reranked_movies, query)
    logging.info(f"Answer: {answer}")

    langfuse_context.score_current_trace(
        name="user_thumbs",
        value=ThumbScore.THUMB_UP.name,
        data_type="CATEGORICAL",
        config_id="cm9sxq8hj00ubad07sz0sffdp",
    )
    langfuse_context.update_current_trace(input=query, output=answer)


if __name__ == "__main__":
    main()
