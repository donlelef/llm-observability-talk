# Observability for the GenAI Era

Code snippets for the talk "Observability for the GenAI Era", first held at PyCon & PyData DE 2025.

# Getting Started

Make sure your have [uv](https://docs.astral.sh/uv/) installed.
Then, install the dependencies:

```bash
uv sync
```

Next, copy the `.env.example` file to `.env` and fill in the required values:

```bash
cp .env.example .env
```

Finally, you can run the scripts:

```bash
uv run --env-file .env src/create_movie_table.py
uv run --env-file .env src/run_rag_query_with_langfuse.py
uv run --env-file .env src/run_rag_query_with_phoenix.py
```

# Slides
The slides for the talk are available on [Canva](https://www.canva.com/design/DAGlZuDWVOg/1hqOmbegW5SImGz-SapttQ/view?utm_content=DAGlZuDWVOg&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hcda31658d1).
