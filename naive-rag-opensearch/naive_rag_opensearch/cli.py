from __future__ import annotations

import argparse
import json

from naive_rag_opensearch.rag import answer_question, index_documents, retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Naive RAG com OpenSearch")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Indexa ficheiros .txt no OpenSearch")
    p_index.add_argument("--input-dir", required=True, help="Pasta com .txt")

    p_search = sub.add_parser("search", help="Busca chunks por similaridade")
    p_search.add_argument("--question", required=True, help="Pergunta de busca")
    p_search.add_argument("--k", type=int, default=5, help="Top-K")

    p_ask = sub.add_parser("ask", help="Executa pipeline Naive RAG")
    p_ask.add_argument("--question", required=True, help="Pergunta")
    p_ask.add_argument("--k", type=int, default=5, help="Top-K")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        total = index_documents(args.input_dir)
        print(f"Total de chunks indexados: {total}")
        return

    if args.command == "search":
        docs = retrieve(args.question, k=args.k)
        print(json.dumps(docs, ensure_ascii=True, indent=2))
        return

    if args.command == "ask":
        out = answer_question(args.question, k=args.k)
        print(out["answer"])
        print("\n--- Contextos recuperados ---")
        print(json.dumps(out["contexts"], ensure_ascii=True, indent=2))
        return


if __name__ == "__main__":
    main()

