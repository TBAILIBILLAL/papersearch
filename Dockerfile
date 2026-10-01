FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY papersearch ./papersearch
RUN pip install --no-cache-dir ".[postgres]"

# the bundled sample is loaded on the first start if the database is empty
COPY data/sample.jsonl ./data/sample.jsonl

EXPOSE 8000
CMD ["papersearch", "serve", "--host", "0.0.0.0", "--port", "8000"]
