from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from opensearchpy import OpenSearch, helpers
from dotenv import load_dotenv


@dataclass
class RAGSettings:
    opensearch_host: str = os.getenv("OPENSEARCH_HOST", "localhost")
    opensearch_port: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    opensearch_user: str = os.getenv("OPENSEARCH_USERNAME", "admin")
    opensearch_password: str = os.getenv("OPENSEARCH_PASSWORD", "admin")
    opensearch_index: str = os.getenv("OPENSEARCH_INDEX", "naive_rag_chunks")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    chunk_size: int = int(os.getenv("NAIVE_RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("NAIVE_RAG_CHUNK_OVERLAP", "150"))


def build_clients(settings: RAGSettings) -> tuple[OpenSearch, OpenAI]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY nao definida no ambiente.")

    os_client = OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30,
    )
    llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return os_client, llm_client


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size // 5, 0)

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def _iter_txt_files(input_dir: Path) -> Iterable[Path]:
    for fp in sorted(input_dir.rglob("*.txt")):
        if fp.is_file():
            yield fp


def _embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return vectors


def _ensure_index(client: OpenSearch, index_name: str, vector_dim: int) -> None:
    if client.indices.exists(index=index_name):
        return
    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {"type": "knn_vector", "dimension": vector_dim},
            }
        },
    }
    client.indices.create(index=index_name, body=body)


def index_documents(input_dir: str | Path) -> int:
    load_dotenv()
    settings = RAGSettings()
    os_client, llm_client = build_clients(settings)

    source_dir = Path(input_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Pasta de entrada inexistente: {source_dir}")

    docs: list[dict] = []
    texts_to_embed: list[str] = []
    for fp in _iter_txt_files(source_dir):
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(raw, settings.chunk_size, settings.chunk_overlap)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{fp.name}::chunk-{idx}"
            docs.append({"chunk_id": chunk_id, "source": str(fp.relative_to(source_dir)), "text": chunk})
            texts_to_embed.append(chunk)

    if not docs:
        return 0

    embeddings = _embed_texts(llm_client, settings.embedding_model, texts_to_embed)
    vector_dim = len(embeddings[0])
    _ensure_index(os_client, settings.opensearch_index, vector_dim)

    actions = []
    for doc, vector in zip(docs, embeddings):
        actions.append(
            {
                "_op_type": "index",
                "_index": settings.opensearch_index,
                "_id": doc["chunk_id"],
                "_source": {**doc, "embedding": vector},
            }
        )

    helpers.bulk(os_client, actions, refresh=True)
    return len(actions)


def retrieve(question: str, k: int = 5) -> list[dict]:
    load_dotenv()
    settings = RAGSettings()
    os_client, llm_client = build_clients(settings)
    q_vec = _embed_texts(llm_client, settings.embedding_model, [question])[0]
    body = {"size": k, "query": {"knn": {"embedding": {"vector": q_vec, "k": k}}}}
    res = os_client.search(index=settings.opensearch_index, body=body)
    hits = res.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source", {})
        out.append(
            {
                "chunk_id": src.get("chunk_id", ""),
                "source": src.get("source", ""),
                "text": src.get("text", ""),
                "score": h.get("_score", 0.0),
            }
        )
    return out


def answer_question(question: str, k: int = 5) -> dict:
    load_dotenv()
    settings = RAGSettings()
    _, llm_client = build_clients(settings)
    contexts = retrieve(question, k=k)
    context_text = "\n\n".join(
        [f"[{c['source']} | {c['chunk_id']}]\n{c['text']}" for c in contexts]
    )
    prompt = (
        "Responda em portugues usando somente o contexto abaixo. "
        "Se nao houver informacao suficiente, diga que nao encontrou.\n\n"
        f"Contexto:\n{context_text}\n\nPergunta: {question}"
    )
    completion = llm_client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = completion.choices[0].message.content or ""
    return {"answer": answer.strip(), "contexts": contexts}

