"""Decompose chain: break question into subqueries (e.g. for similarity + Cypher)."""

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graphrag.llm_factory import create_chat_llm


class SubQuery(BaseModel):
    """A single subquery from decomposition."""

    sub_query: str = Field(description="A clear sub-question or search query.")


class DecomposedQueries(BaseModel):
    """Output of decomposition: list of subqueries."""

    subqueries: List[SubQuery] = Field(description="List of 1 to 3 subqueries.")


_llm = create_chat_llm(temperature=0)
_structured_llm = _llm.with_structured_output(DecomposedQueries)

DECOMPOSE_SYSTEM = """You are an expert at breaking down user questions into sub-queries for a hybrid search system.
Given a question, produce 1 to 3 subqueries that together help answer it:
- One subquery can be for semantic/similarity search (e.g. find relevant text chunks).
- Another can be for a graph/structured query (e.g. find entities and relationships).
Return only the list of subqueries."""

_decompose_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", DECOMPOSE_SYSTEM),
        ("human", "{question}"),
    ]
)

decompose_chain = _decompose_prompt | _structured_llm
