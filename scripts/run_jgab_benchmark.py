"""
Benchmark: GraphRAG (fluxo graphrag-neo4j-langchain) vs RAG ingênuo (só vetor TextUnit).

Métricas:
  - Recuperação: recall@5 e precision@5 sobre TextUnits (ids alinhados ao pipeline de chunking).
  - Geração: F1 por tokens (sobreposição de tokens entre resposta e gabarito).

Pré-requisitos:
  - Indexar o corpus com o mesmo ficheiro em Neo4j (ex.: python scripts/run_indexing.py -i <pasta_com_txt>).
  - .env no graphrag-neo4j-langchain com NEO4J_* e OPENAI_API_KEY.

Uso (na raiz do projeto benchmarking-graphrag):
  python scripts/run_jgab_benchmark.py
  python scripts/run_jgab_benchmark.py --output-csv outputs/jgab_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _setup_paths_and_env() -> Path:
    root = _project_root()
    gr = root / "graphrag-neo4j-langchain"
    src = gr / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Pasta graphrag esperada em: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    env = gr / ".env"
    if env.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env)
        except ImportError:
            pass
    return gr


@dataclass
class QARow:
    question: str
    reference_answer: str
    hop_type: str
    gold_chunk_indices: list[int]


def _normalize_tokens(text: str) -> list[str]:
    t = (text or "").lower()
    t = re.sub(r"[^\w\s\u00C0-\u024F]", " ", t, flags=re.UNICODE)
    return [x for x in t.split() if x]


def token_f1(reference: str, predicted: str) -> float:
    """F1 por tokens (estilo avaliação aberta de QA)."""
    ref = _normalize_tokens(reference)
    hyp = _normalize_tokens(predicted)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    ref_set = ref
    hyp_set = hyp
    cr = Counter(ref_set)
    ch = Counter(hyp_set)
    overlap = sum((cr & ch).values())
    prec = overlap / max(len(hyp_set), 1)
    rec = overlap / max(len(ref_set), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    top = retrieved_ids[:k]
    hits = sum(1 for x in top if x in gold_ids)
    if not gold_ids:
        return 1.0
    return hits / len(gold_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    top = retrieved_ids[:k]
    if k <= 0:
        return 0.0
    hits = sum(1 for x in top if x in gold_ids)
    return hits / k


def load_qa_csv(path: Path) -> list[QARow]:
    rows: list[QARow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        req = {"question", "reference_answer", "hop_type", "relevant_chunk_indices"}
        miss = req - set(reader.fieldnames or [])
        if miss:
            raise ValueError(f"CSV incompleto. Faltam colunas: {sorted(miss)}")
        for r in reader:
            q = (r.get("question") or "").strip()
            ra = (r.get("reference_answer") or "").strip()
            hop = (r.get("hop_type") or "").strip()
            idx_raw = (r.get("relevant_chunk_indices") or "").strip()
            if not q or not ra:
                continue
            indices = []
            for x in idx_raw.split(","):
                xs = x.strip()
                if xs.isdigit():
                    indices.append(int(xs))
            rows.append(QARow(question=q, reference_answer=ra, hop_type=hop, gold_chunk_indices=indices))
    if not rows:
        raise ValueError("Nenhuma linha válida no CSV de QA.")
    return rows


def build_chunk_index_to_tu_id(doc_path: Path) -> tuple[str, dict[int, str]]:
    """Replica ids TextUnit: {resolved_source}_{chunk_index}."""
    _setup_paths_and_env()
    from langchain_core.documents import Document
    from graphrag.indexing.load_and_chunk import chunk_documents

    doc_path = doc_path.resolve()
    text = doc_path.read_text(encoding="utf-8")
    doc = Document(page_content=text, metadata={"source": str(doc_path)})
    chunks = chunk_documents([doc])
    doc_id = str(doc_path)
    mapping = {i: f"{doc_id}_{i}" for i in range(len(chunks))}
    return doc_id, mapping


def naive_retrieve_top5_text_unit_ids(question: str) -> tuple[list[str], list[str]]:
    """Vetor Neo4j TextUnit, k=5. Devolve (ids, textos)."""
    from graphrag.store.vector_index import get_vector_index_text_units

    store = get_vector_index_text_units()
    if store is None:
        raise RuntimeError("Índice vetorial de TextUnits indisponível. Execute o pipeline de embed.")
    docs = store.similarity_search(question.strip(), k=5)
    ids: list[str] = []
    texts: list[str] = []
    for d in docs:
        md = getattr(d, "metadata", None) or {}
        tid = md.get("id")
        if tid is None:
            tid = md.get("t.id")
        ids.append(str(tid) if tid is not None else "")
        texts.append(getattr(d, "page_content", "") or "")
    return ids, texts


def graphrag_top5_text_unit_ids(state: dict) -> list[str]:
    """Primeiros 5 TextUnit ids únicos na ordem do contexto empacotado."""
    out: list[str] = []
    for doc in state.get("context_docs") or []:
        if not isinstance(doc, dict):
            continue
        md = doc.get("metadata") or {}
        if not isinstance(md, dict):
            continue
        kind = str(md.get("kind") or "")
        if not kind.startswith("text_unit"):
            continue
        tid = md.get("text_unit_id")
        if not tid:
            continue
        s = str(tid)
        if s not in out:
            out.append(s)
        if len(out) >= 5:
            break
    return out


def run_graphrag_full(question: str, thread_tag: str) -> dict:
    from graphrag.graph.query_graph import get_compiled_graph

    compiled = get_compiled_graph()
    cfg = {"configurable": {"thread_id": f"jgab-bench-{thread_tag}"}}
    return compiled.invoke({"question": question}, config=cfg)


def naive_generate_answer(question: str, context_chunks: list[str]) -> str:
    from graphrag.config import OPENAI_API_KEY, LLM_MODEL
    from graphrag.monitoring.token_cost import tracked_chat_openai
    from graphrag.prompts.synthesis import SYNTHESIS_PROMPT

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não definida.")
    ctx = "\n\n".join(context_chunks) if context_chunks else "(sem contexto)"
    llm = tracked_chat_openai(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    chain = SYNTHESIS_PROMPT | llm
    msg = chain.invoke({"context": ctx, "question": question})
    return getattr(msg, "content", str(msg)) or ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark GraphRAG vs RAG ingênuo (JGab21-285).")
    p.add_argument(
        "--doc-path",
        type=Path,
        default=_project_root() / "docs" / "JGab21-285.txt",
        help="Ficheiro .txt indexado (mesmo caminho absoluto que no Neo4j).",
    )
    p.add_argument(
        "--qa-csv",
        type=Path,
        default=_project_root() / "docs" / "jgab_benchmark_qa.csv",
        help="CSV com question,reference_answer,hop_type,relevant_chunk_indices.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=_project_root() / "outputs" / "jgab_benchmark_runs.csv",
        help="CSV com uma linha por pergunta e por sistema.",
    )
    p.add_argument("--k", type=int, default=5, help="k para recall@k e precision@k.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_paths_and_env()

    from graphrag.config import OPENAI_API_KEY

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não configurada (graphrag-neo4j-langchain/.env).")

    doc_path = args.doc_path.resolve()
    if not doc_path.is_file():
        raise FileNotFoundError(doc_path)

    _, idx_to_tu = build_chunk_index_to_tu_id(doc_path)
    qa_rows = load_qa_csv(args.qa_csv.resolve())
    k = max(1, int(args.k))

    out_rows: list[dict] = []
    sums = {
        "naive": {"recall": 0.0, "precision": 0.0, "f1": 0.0, "n": 0},
        "graphrag": {"recall": 0.0, "precision": 0.0, "f1": 0.0, "n": 0},
    }

    for i, row in enumerate(qa_rows):
        gidx = sorted(set(row.gold_chunk_indices))
        missing = set(gidx) - set(idx_to_tu.keys())
        if missing:
            raise ValueError(f"Índices de chunk inválidos para a pergunta {i+1}: {sorted(missing)}")
        gold_ids = {idx_to_tu[j] for j in gidx}

        tag = str(abs(hash(row.question)) % 10_000_000)

        # Naive
        n_ids, n_texts = naive_retrieve_top5_text_unit_ids(row.question)
        n_rec = recall_at_k(n_ids, gold_ids, k)
        n_prec = precision_at_k(n_ids, gold_ids, k)
        n_ans = naive_generate_answer(row.question, n_texts)
        n_f1 = token_f1(row.reference_answer, n_ans)

        # GraphRAG
        g_state = run_graphrag_full(row.question, tag)
        g_ids = graphrag_top5_text_unit_ids(g_state)
        g_ans = (g_state.get("final_answer") or "").strip()
        g_rec = recall_at_k(g_ids, gold_ids, k)
        g_prec = precision_at_k(g_ids, gold_ids, k)
        g_f1 = token_f1(row.reference_answer, g_ans)

        for sys_name, rec, prec, f1, ans, rids in [
            ("naive_rag", n_rec, n_prec, n_f1, n_ans, "|".join(n_ids)),
            ("graphrag", g_rec, g_prec, g_f1, g_ans, "|".join(g_ids)),
        ]:
            sums_key = "naive" if sys_name == "naive_rag" else "graphrag"
            s = sums[sums_key]
            s["recall"] += rec
            s["precision"] += prec
            s["f1"] += f1
            s["n"] += 1

            out_rows.append(
                {
                    "system": sys_name,
                    "question_index": i + 1,
                    "hop_type": row.hop_type,
                    "question": row.question,
                    "reference_answer": row.reference_answer,
                    "generated_answer": ans,
                    f"recall_at_{k}": round(rec, 6),
                    f"precision_at_{k}": round(prec, 6),
                    "token_f1": round(f1, 6),
                    "retrieved_text_unit_ids": rids,
                    "gold_text_unit_ids": "|".join(sorted(gold_ids)),
                }
            )

        print(f"[{i+1}/{len(qa_rows)}] {row.hop_type} OK (naive R@k={n_rec:.3f} G R@k={g_rec:.3f})")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(out_rows)

    nq = len(qa_rows)
    print("\n--- Médias macro (por pergunta) ---")
    for label, key in [("RAG ingênuo", "naive"), ("GraphRAG", "graphrag")]:
        s = sums[key]
        if s["n"]:
            print(
                f"{label}: recall@{k}={s['recall']/s['n']:.4f} "
                f"precision@{k}={s['precision']/s['n']:.4f} "
                f"token-F1={s['f1']/s['n']:.4f}"
            )
    print(f"\nResultados por linha: {args.output_csv}")


if __name__ == "__main__":
    main()
