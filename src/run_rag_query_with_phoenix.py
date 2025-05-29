import logging
import uuid

import lancedb
from openai import OpenAI
from openinference.instrumentation import using_session, using_user
from opentelemetry.trace import format_span_id, get_current_span, StatusCode
from phoenix.client import Client
from phoenix.otel import register

from embeddings import Movie
from paths import LANCEDB_URI
from rag import build_hyde_query, run_semantic_query, run_reranking, answer_query_from_context
from scores import ThumbScore


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    logging.info("Registering Phoenix as OTEL provider...")
    tracer_provider = register(batch=True, auto_instrument=True, set_global_tracer_provider=False)
    tracer = tracer_provider.get_tracer(__name__)
    logging.info("Registration completed.")

    session_id = uuid.uuid4().hex
    user_id = "AnonimizedLele"
    logging.info(f"Session ID: {session_id}, user ID: {user_id}")

    openai_client = OpenAI()
    phoenix_client = Client()

    with (
        tracer.start_as_current_span("run_rag_query_with_phoenix", openinference_span_kind="chain") as span,
        using_session(session_id),
        using_user(user_id),
    ):
        query = "I would like to watch a movie with dragons"

        logging.info(f"Query: {query}")
        logging.info("Building hyde query...")
        hyde_query = build_hyde_query(openai_client, query)
        logging.info(f"Hyde query: {hyde_query}...")

        logging.info("Connecting to LanceDB...")
        db = lancedb.connect(LANCEDB_URI)
        logging.info("Running queries...")
        res: list[Movie] = run_semantic_query(db, query)
        hyde_res: list[Movie] = run_semantic_query(db, hyde_query)
        logging.info(f"Length for original query {len(res)}, length of hyde query: {len(hyde_res)}")

        logging.info("Re-ranking results...")
        reranked_movies = run_reranking(openai_client, res + hyde_res, query)
        logging.info(f"Reranked movies: {reranked_movies}")

        logging.info("Answering query...")
        answer = answer_query_from_context(openai_client, reranked_movies, query)
        logging.info(f"Answer: {answer}")

        logging.info("Sending feedback to answer...")
        phoenix_client.annotations.add_span_annotation(
            span_id=format_span_id(get_current_span().get_span_context().span_id),
            annotation_name="feedback",
            annotator_kind="HUMAN",
            label=ThumbScore.THUMB_UP.name,
            score=ThumbScore.THUMB_UP.value,
        )
        span.set_input(query)
        span.set_output(answer)
        span.set_status(StatusCode.OK)


if __name__ == "__main__":
    main()
