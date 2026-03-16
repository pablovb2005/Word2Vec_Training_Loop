FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN python -m pip install --upgrade pip \
    && pip install -e .

ENTRYPOINT ["python", "-m", "word2vec"]
CMD ["--epochs", "20", "--queries", "word,vectors,tiny,context"]
