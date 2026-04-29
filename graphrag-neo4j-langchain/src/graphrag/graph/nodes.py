"""LangGraph nodes: router, decompose, local_retrieve, graph_qa, synthesize, global_stub."""

from typing import Any

from graphrag.state import GraphRAGState
from graphrag.chains.router import route_chain
from graphrag.chains.decompose import decompose_chain
from graphrag.retrieval.local_search import build_local_search_context
from graphrag.retrieval.global_search import fetch_global_community_reports, global_search_map_reduce
from graphrag.chains.graph_qa import get_graph_qa_chain
from graphrag.prompts.cypher import create_cypher_prompt, create_cypher_prompt_with_context
from graphrag.prompts.synthesis import SYNTHESIS_PROMPT
from graphrag.config import LOCAL_SYNTH_CONTEXT_DOC_CAP
from graphrag.llm_factory import create_chat_llm


def _subquery_text(item: Any) -> str:
    if item is None:
        return ""
    val = getattr(item, "sub_query", None)
    if val is None and isinstance(item, dict):
        val = item.get("sub_query")
    if val is None:
        return ""
    return str(val).strip()


def _extract_generated_cypher(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    steps = result.get("intermediate_steps")
    if not isinstance(steps, list):
        return ""
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        query = step.get("query")
        if query:
            return str(query).strip()
    return ""


def _repair_instruction(question: str, error_text: str) -> str:
    return (
        f"{question}\n\n"
        "Previous Cypher attempt failed with this Neo4j error:\n"
        f"{error_text}\n\n"
        "Generate a corrected Cypher query using only the provided schema.\n"
        "Keep result shape equivalent to the original intent.\n"
        "Return only the corrected Cypher query."
    )


def router_node(state: GraphRAGState) -> dict:
    """Classify question as local or global."""
    decision = route_chain.invoke({"question": state["question"]})
    print(f"Decision: {decision.search_type}")
    return {"search_type": decision.search_type}


def decompose_node(state: GraphRAGState) -> dict:
    """Break question into subqueries."""
    result = decompose_chain.invoke({"question": state["question"]})
    print(f"Subqueries: {result.subqueries}")
    return {"subqueries": result.subqueries}


def local_retrieve_node(state: GraphRAGState) -> dict:
    """GraphRAG-style local search: match entities, fan-out to text units, rels, neighbors, community reports, + vector text."""
    context_docs, seed_entities = build_local_search_context(state)
    print(f"Local retrieve: {len(seed_entities)} seed entities, {len(context_docs)} context chunks")
    return {"context_docs": context_docs, "seed_entities": seed_entities}


def graph_qa_node(state: GraphRAGState) -> dict:
    """Run Cypher QA; fill cypher_result."""
    subqueries = state.get("subqueries") or []
    query = _subquery_text(subqueries[0]) if subqueries else ""
    if not query:
        query = state["question"]

    use_ctx = bool(state.get("context_docs") or state.get("seed_entities"))
    cypher_prompt = create_cypher_prompt_with_context(state) if use_ctx else create_cypher_prompt()
    chain = get_graph_qa_chain(cypher_prompt=cypher_prompt)
    current_query = query
    attempts = 3
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            result = chain.invoke({"query": current_query})
            generated_cypher = _extract_generated_cypher(result)
            if generated_cypher:
                print(f"Graph QA Cypher (attempt {attempt}): {generated_cypher}")
            return {"cypher_result": result, "cypher_error": ""}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"Graph QA failed on attempt {attempt}/{attempts}: {last_error}")
            if attempt < attempts:
                current_query = _repair_instruction(query, last_error)
                continue

    # Keep the pipeline alive even when Cypher generation/execution fails.
    return {
        "cypher_result": "",
        "cypher_error": (
            "Cypher unavailable after retries; continuing with local context only. "
            + last_error
        ),
    }


def synthesize_node(state: GraphRAGState) -> dict:
    """Combine context_docs + cypher_result into final_answer."""
    parts = []
    for doc in (state.get("context_docs") or [])[:LOCAL_SYNTH_CONTEXT_DOC_CAP]:
        content = doc.get("page_content", doc) if isinstance(doc, dict) else str(doc)
        parts.append(content)
    cypher_result = state.get("cypher_result")
    if cypher_result:
        parts.append(str(cypher_result))
    cypher_error = (state.get("cypher_error") or "").strip()
    if cypher_error:
        parts.append(f"[Cypher fallback] {cypher_error}")
    context = "\n\n".join(parts) if parts else "No relevant context found."
    llm = create_chat_llm(temperature=0)
    chain = SYNTHESIS_PROMPT | llm
    answer = chain.invoke({"context": context, "question": state["question"]})
    if hasattr(answer, "content"):
        answer = answer.content
    return {"final_answer": answer}


def global_stub_node(state: GraphRAGState) -> dict:
    """Fallback when no community reports index: return message."""
    return {"final_answer": "Busca global ainda não implementada. Indexe documentos e execute o pipeline de indexação."}


def global_retrieve_node(state: GraphRAGState) -> dict:
    """Recupera pool amplo de Community Reports (vetor) e embaralha para o map-reduce global."""
    reports = fetch_global_community_reports(state["question"])
    print(f"Global retrieve: {len(reports)} community reports (pooled + shuffled)")
    return {"community_reports": reports}


def global_synthesize_node(state: GraphRAGState) -> dict:
    """Global search estilo GraphRAG: map (pontos pontuados por lote) → reduce (resposta final)."""
    reports = state.get("community_reports") or []
    answer = global_search_map_reduce(state["question"], reports)
    return {"final_answer": answer}
