import logging

import lancedb
import polars as pl

from src.embeddings import Movie
from src.paths import DATA_PATH, LANCEDB_URI


def main():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

    logging.info(f"Reading data from {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH)
    logging.info("Connecting to LanceDB...")
    db = lancedb.connect(LANCEDB_URI)

    logging.info("Creating table...")
    table = db.create_table("words", schema=Movie, mode="overwrite")
    logging.info("Adding data to table...")
    batch_size = 2_000
    for i in range(0, df.height, batch_size):
        logging.info(f"Adding rows {i} to {i + batch_size}...")
        table.add(df[i:i + batch_size], mode='append')

    logging.info("Running query...")
    res: Movie = table.search("I would like a movie with dragons.").limit(1).to_pydantic(Movie)[0]
    logging.info(f"Query results: \n Title: {res.title} \n Overview: {res.overview}")


if __name__ == '__main__':
    main()
