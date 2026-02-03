# modules/query_agent.py

import logging
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from app.modules.llm_handler import LLMHandler
from app.modules.graph_store import GraphStore
from app.modules.prompts.query_prompts import (
    MENTION_PROMPT,
    EXPLAIN_PROMPT,
    CYPHER_PROMPT,
    ANSWER_PROMPT,
)

from app.modules.file_utils import log_json, log_cypher_queries


class CypherParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1)
    value: str


class CypherQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cypher: str = Field(..., min_length=1)
    params: list[CypherParam] = Field(default_factory=list)
    reason: str | None = None


class CypherQueryList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[CypherQuery] = Field(default_factory=list)


class QueryAgent:
    def __init__(self, llm: LLMHandler, graph_store: GraphStore):
        self.llm = llm
        self.graph_store = graph_store
        self.log = logging.getLogger(__name__)

    # ---------------------------
    # 1. Extract mentions
    # ---------------------------
    def extract_mentions(self, question: str):
        schema_text = self.graph_store.get_supply_chain_ontology_text()
        prompt = MENTION_PROMPT.format(SCHEMA=schema_text) + f'\n\nQuestion:\n"{question}"\n'
        self.log.info(f"Extract mentions prompt: {prompt}")

        result = self.llm.call_json(prompt)
        self.log.info(f"Extract mentions raw result: {result}")
        self.log.info(f"type: {type(result)}")

        # CASE 1: Already a list
        if isinstance(result, list):
            return result

        # CASE 2: The model wrapped it in an object: {"entities": [...]}
        if isinstance(result, dict) and "entities" in result:
            ents = result.get("entities", [])
            if isinstance(ents, list):
                return ents

        # CASE 3: Totally unexpected / fail-safe fallback
        return []

    # ---------------------------
    # 2. Resolve mentions → nodes
    # ---------------------------
    def resolve_all(self, mentions):
        resolved = []
        for term in mentions:
            node = self.graph_store.resolve_entity(term)
            if node:
                resolved.append({"mention": term, "entity": node})
            else:
                resolved.append({"mention": term, "entity": None})
        return resolved

    # ---------------------------
    # 3. Relationship search
    # ---------------------------
    def find_connections(self, resolved):
        # Take the first two resolved entities
        entities = [r["entity"] for r in resolved if r["entity"]]
        if len(entities) < 2:
            return None

        e1, e2 = entities[0], entities[1]

        rel_result = self.graph_store.find_relationships(e1["id"], e2["id"])

        return {
            "entity1": e1,
            "entity2": e2,
            "relationships": rel_result["rows"],
            "cypher": rel_result["cypher"],
            "params": rel_result["params"],
        }


    # ---------------------------
    # 4. Explanation
    # ---------------------------
    def explain(self, question: str, connection, scenario_text: str | None = None):
        ent_txt = str(connection["entity1"]) + "\n" + str(connection["entity2"])
        rel_txt = str(connection["relationships"])

        schema_text = self.graph_store.get_supply_chain_ontology_text()
        prompt = EXPLAIN_PROMPT.format(
            SCHEMA=schema_text,
            ENTITIES=ent_txt,
            RELATIONS=rel_txt,
            QUESTION=question,
            SCENARIO=scenario_text or "No scenario constraints applied.",
        )
        return self.llm.call(prompt)

    # ---------------------------
    # MAIN ENTRYPOINT
    # ---------------------------
    def ask(self, question: str, scenario_text: str | None = None):
        self.log.info("Reasoning QueryAgent: starting")

        # 1. Extract mentions
        mentions = self.extract_mentions(question)
        self.log.info(f"Extracted mentions: {mentions}")

        # 2. Resolve mentions
        resolved = self.resolve_all(mentions)
        self.log.info(f"Resolved entities: {resolved}")

        # 3. Try to connect the first two resolved entities
        connection = self.find_connections(resolved)
        self.log.info(f"Found connection: {connection}")

        # If no resolvable connection
        if (not connection) or (not connection.get("relationships")):
            return {
                "answer": "No explicit connection found in the graph.",
                "mentions": mentions,
                "resolved": resolved,
                "relationships": [],
                "raw": [],
                "cypher": None
            }

        # 4. Build explanation
        explanation = self.explain(question, connection, scenario_text)

        return {
            "answer": explanation,
            "mentions": mentions,
            "resolved": resolved,
            "relationships": connection["relationships"],
            "raw": connection["relationships"],
            "cypher": connection["cypher"],
            "params": connection["params"]
        }

    # ---------------------------
    # Cypher-first GraphRAG
    # ---------------------------
    def generate_cypher_queries(self, question: str, scenario_text: str | None = None):
        schema_text = self.graph_store.get_supply_chain_ontology_text()
        prompt = CYPHER_PROMPT.format(
            SCHEMA=schema_text,
            SCENARIO=scenario_text or "No scenario constraints applied.",
            QUESTION=question,
        )
        log_json("cypher_prompt.json", {"prompt": prompt})
        result = self.llm.call_schema_prompt(prompt, CypherQueryList)
        log_json("cypher_prompt_result.json", {"result": result})

        if isinstance(result, CypherQueryList):
            items = result.queries
        elif isinstance(result, dict):
            items = CypherQueryList.model_validate(result).queries
        else:
            items = []

        normalized = [
            {
                "cypher": item.cypher.strip(),
                "params": {param.key: param.value for param in (item.params or [])},
                "reason": item.reason,
            }
            for item in items
            if item.cypher and item.cypher.strip()
        ]
        log_cypher_queries("cypher_queries.cypher", normalized)
        return normalized

    def execute_cypher_queries(self, queries):
        results = []
        for query in queries:
            cypher = query["cypher"]
            params = query.get("params") or {}
            rows = self.graph_store.run_cypher(cypher, params)
            results.append(
                {
                    "cypher": cypher,
                    "params": params,
                    "reason": query.get("reason"),
                    "row_count": len(rows),
                    "rows": rows[:50],
                }
            )
        return results

    def answer_from_cypher(self, question: str, scenario_text: str | None, results):
        schema_text = self.graph_store.get_supply_chain_ontology_text()
        prompt = ANSWER_PROMPT.format(
            SCHEMA=schema_text,
            SCENARIO=scenario_text or "No scenario constraints applied.",
            QUESTION=question,
            RESULTS=results,
        )
        return self.llm.call(prompt)

    def ask_cypher(self, question: str, scenario_text: str | None = None):
        queries = self.generate_cypher_queries(question, scenario_text)
        results = self.execute_cypher_queries(queries) if queries else []
        answer = self.answer_from_cypher(question, scenario_text, results)
        return {
            "answer": answer,
            "queries": queries,
            "results": results,
        }
