"""Graph Cypher QA chain over Neo4j."""

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from graphrag.llm_factory import create_chat_llm
from graphrag.store.neo4j_graph import get_neo4j_graph
from graphrag.prompts.cypher import create_cypher_prompt


def get_graph_qa_chain(cypher_prompt=None):
    """Return a GraphCypherQAChain. Optionally pass a custom cypher_prompt."""
    graph = get_neo4j_graph()
    llm = create_chat_llm(temperature=0)
    prompt = cypher_prompt or create_cypher_prompt()
    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=prompt,
        validate_cypher=True,
        return_direct=True,
        return_intermediate_steps=True,
        verbose=False,
        allow_dangerous_requests=True
    )
