# Naive RAG com OpenSearch (Modulo Independente)

Modulo independente para montar uma arquitetura Naive RAG usando OpenSearch como armazenamento vetorial.

## O que este modulo faz

- Le ficheiros `.txt` de uma pasta
- Divide em chunks
- Gera embeddings com OpenAI
- Indexa no OpenSearch (`knn_vector`)
- Recupera top-k por similaridade vetorial
- Gera resposta final (RAG) com OpenAI

## Requisitos

- OpenSearch ativo (via `docker compose`)
- Python 3.10+
- `OPENAI_API_KEY` definida

## Instalacao

```bash
cd naive-rag-opensearch
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

## Uso

### 1) Indexar documentos

```bash
naive-rag-opensearch index --input-dir ../pdfs_txt
```

### 2) Buscar chunks

```bash
naive-rag-opensearch search --question "Qual foi a data do incidente?" --k 5
```

### 3) Fazer pergunta (pipeline RAG)

```bash
naive-rag-opensearch ask --question "Qual foi a data do incidente?" --k 5
```

## Notas

- O indice padrao e `naive_rag_chunks`.
- Para forcar outro indice, ajuste `OPENSEARCH_INDEX` no `.env`.
- O modulo e intencionalmente simples para servir como baseline Naive RAG.

