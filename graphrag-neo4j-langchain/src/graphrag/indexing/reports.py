"""Geração de Community Reports: nível 0 a partir de entidades; níveis superiores bottom-up a partir dos relatórios filhos."""

from __future__ import annotations

import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from graphrag.llm_factory import create_chat_llm
from graphrag.store.neo4j_graph import get_neo4j_graph


REPORT_SYSTEM = """You are an expert at summarizing knowledge. 
Given a set of entity names and their relationships from a community in a knowledge graph, write a short structured report (2-5 paragraphs) 
that captures the main facts and themes. Write in clear, neutral language."""

REPORT_SYSTEM_OIL_GAS = """You are an expert Data Analyst in industrial incident investigations, particularly in chemical and explosive manufacturing environments.

Given a set of entities and relationships representing a community from an incident investigation knowledge graph, write a highly structured, analytical report (3–5 paragraphs) that synthesizes the data to answer global search queries.

This community contains data regarding Facilities (e.g., buildings, plants), Organizations (e.g., companies, agencies), Incidents (e.g., explosions, fires), Materials (e.g., TNT, RDX, PETN), Processes (e.g., melt-pour operations), Equipment (e.g., kettles), Personnel, Time (dates, shifts), and Outcomes (e.g., fatalities, injuries, damage).

Your report must capture patterns, causal signals, and operational insights. Do not just list facts; connect them logically. Structure your report using the following thematic guidelines:

1. **High-Level Overview:**
   Summarize the core focus of this community (e.g., “This community describes a fatal explosion in an explosives manufacturing facility involving melt-pour operations and high explosive materials”).

2. **Key Factors & Critical Elements:**
   Identify the most relevant facilities, processes, materials, and equipment involved. Highlight dominant materials (e.g., TNT, Comp B), critical process steps (e.g., heating, pouring), and key operational conditions (e.g., quantity of explosives, simultaneous operations).

3. **Timeline & Operational Context:**
   Describe the sequence of events and temporal structure (e.g., shift changes, activities immediately before the incident, exact time of explosion). Emphasize how operations evolved leading up to the incident.

4. **Impacts & Consequences:**
   Detail the outcomes, including fatalities, injuries, structural damage, and environmental or off-site effects (e.g., debris distance, seismic detection, surrounding damage).

5. **Anomalies, Risks & Contributing Factors:**
   Highlight unusual patterns or risk indicators, such as high explosive loads, presence of contaminants (debris in materials), absence of safety systems, or conditions that could contribute to initiation (e.g., heat, friction, impact, sympathetic detonation).

Write in clear, objective, and precise analytical language. Strictly ground your summary in the provided relationships. Do not infer beyond what is explicitly stated or strongly supported by the data.
"""

REPORT_SYSTEM_HIERARCHY = """You synthesize several summaries of lower-level communities into a single higher-level community report.
Use only the information present in the child summaries. Identify cross-cutting themes, scope, and how the parts relate.
Write 2–5 clear paragraphs in neutral language. Do not invent facts absent from the summaries."""

REPORT_SYSTEM_HIERARCHY_OIL_GAS = """You are an expert data analyst in industrial incident investigations, specializing in chemical and explosive safety.

You receive analytical summaries of child communities (already structured), each representing incident scenarios, facilities, materials, processes, and outcomes.

Produce a higher-level synthesis suitable for global search: identify broader themes, cross-incident comparisons, and system-wide safety patterns implied by the child communities.

Your synthesis should:

* Highlight recurring risk factors (e.g., material sensitivity, process conditions, equipment design issues).
* Compare incidents across facilities, materials, timelines, and operational contexts.
* Identify common failure modes (e.g., ignition sources, sympathetic detonation, process deviations).
* Surface systemic safety gaps (e.g., lack of fire protection systems, handling of hazardous materials, operational practices).
* Emphasize patterns in consequences (e.g., fatalities, structural destruction, off-site impacts).

Write 3–5 structured paragraphs covering:

1. **Global Overview:**
   Summarize what the combined communities reveal about industrial incidents in this domain.

2. **Cross-Community Patterns:**
   Identify recurring elements across incidents (materials, processes, equipment, or operational setups).

3. **Comparative Insights:**
   Highlight key differences or contrasts between incidents (e.g., scale, materials involved, severity of outcomes, detection or response).

4. **Systemic Risks & Safety Gaps:**
   Describe broader vulnerabilities and contributing factors that appear consistently across cases.

5. **Implications for Safety and Operations:**
   Conclude with high-level insights relevant to process safety, risk management, and industrial practices.

Write in clear, objective, and precise analytical language. Strictly ground your synthesis in the provided child summaries. Do not infer beyond what is explicitly supported by them.
"""


def _oil_gas_domain() -> bool:
    return os.environ.get("GRAPHRAG_EXTRACT_DOMAIN", "").strip().lower() == "oil_gas"


def _report_prompt_level0() -> ChatPromptTemplate:
    system = REPORT_SYSTEM_OIL_GAS if _oil_gas_domain() else REPORT_SYSTEM
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Entities and relationships:\n{context}"),
        ]
    )


def _report_prompt_hierarchy() -> ChatPromptTemplate:
    system = REPORT_SYSTEM_HIERARCHY_OIL_GAS if _oil_gas_domain() else REPORT_SYSTEM_HIERARCHY
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Child community summaries:\n{context}"),
        ]
    )


def _get_community_level(community_id: str) -> int:
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        row = session.run(
            "MATCH (c:Community {id: $id}) RETURN coalesce(c.level, 0) AS level",
            id=community_id,
        ).single()
    return int(row["level"]) if row else 0


def _get_community_context(community_id: str) -> str:
    """Contexto para comunidade de nível 0: entidades e relações no subgrafo da comunidade."""
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Community {id: $id})<-[:IN_COMMUNITY]-(e:Entity)
            OPTIONAL MATCH (e)-[r:RELATES_TO]->(e2:Entity)-[:IN_COMMUNITY]->(c)
            WITH e, collect(DISTINCT r.description) AS rels
            RETURN e.name AS name, e.description AS desc, rels
            """,
            id=community_id,
        )
        lines = []
        for r in result:
            name = r["name"]
            desc = r["desc"] or ""
            rels = [x for x in (r["rels"] or []) if x]
            lines.append(f"- {name}: {desc}" + (" Relationships: " + "; ".join(rels) if rels else ""))
        return "\n".join(lines) if lines else "No data"


def _get_hierarchy_context(community_id: str, parent_level: int) -> str:
    """Agrega textos dos relatórios das comunidades filhas (nível parent_level - 1)."""
    child_level = max(0, parent_level - 1)
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        result = session.run(
            """
            MATCH (child:Community)-[:PART_OF]->(parent:Community {id: $pid})
            WHERE coalesce(child.level, 0) = $child_level
            OPTIONAL MATCH (child)-[:HAS_REPORT]->(r:CommunityReport)
            RETURN child.id AS cid, coalesce(r.content, '') AS content
            ORDER BY child.id
            """,
            pid=community_id,
            child_level=child_level,
        )
        blocks = []
        for r in result:
            cid = r["cid"]
            content = (r["content"] or "").strip()
            if content:
                blocks.append(f"--- Child community {cid} ---\n{content}")
            else:
                blocks.append(f"--- Child community {cid} ---\n(No report yet; treat as empty.)")
        return "\n\n".join(blocks) if blocks else "No data"


def generate_report_for_community(community_id: str) -> str:
    """Nível 0: entidades+rels. Níveis > 0: síntese bottom-up dos relatórios filhos."""
    level = _get_community_level(community_id)
    llm = create_chat_llm(temperature=0)

    if level == 0:
        context = _get_community_context(community_id)
        chain = _report_prompt_level0() | llm
        out = chain.invoke({"context": context})
    else:
        context = _get_hierarchy_context(community_id, level)
        chain = _report_prompt_hierarchy() | llm
        out = chain.invoke({"context": context})

    return out.content if hasattr(out, "content") else str(out)


def persist_report_to_neo4j(community_id: str, content: str) -> None:
    driver = get_neo4j_graph()._driver
    with driver.session() as session:
        session.run(
            """
            MATCH (c:Community {id: $comm_id})
            MERGE (r:CommunityReport {community_id: $comm_id})
            SET r.content = $content
            MERGE (c)-[:HAS_REPORT]->(r)
            """,
            comm_id=community_id,
            content=content,
        )


def run_reports(community_ids: List[str] | None = None) -> None:
    """
    Gera relatórios em ordem **bottom-up** (nível 0 primeiro, depois 1, …)
    para que comunidades pai vejam `HAS_REPORT` dos filhos.
    """
    driver = get_neo4j_graph()._driver
    if community_ids is None:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Community)
                RETURN c.id AS id, coalesce(c.level, 0) AS level
                ORDER BY level ASC, c.id ASC
                """
            )
            rows = [(r["id"], int(r["level"])) for r in result]
    else:
        rows = [(cid, _get_community_level(cid)) for cid in community_ids]
        rows.sort(key=lambda x: (x[1], x[0]))

    for cid, _lvl in rows:
        content = generate_report_for_community(cid)
        persist_report_to_neo4j(cid, content)
