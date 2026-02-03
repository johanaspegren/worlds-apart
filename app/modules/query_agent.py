# modules/query_agent.py

import logging
from modules.llm_handler import LLMHandler
from modules.graph_store import GraphStore
from modules.prompts.query_prompts import MENTION_PROMPT, EXPLAIN_PROMPT


class QueryAgent:
    def __init__(self, llm: LLMHandler, graph_store: GraphStore):
        self.llm = llm
        self.graph_store = graph_store
        self.log = logging.getLogger(__name__)

    # ---------------------------
    # 1. Extract mentions
    # ---------------------------
    def extract_mentions(self, question: str):
        prompt = MENTION_PROMPT + f'\n\nQuestion:\n"{question}"\n'
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
    def explain(self, question: str, connection):
        ent_txt = str(connection["entity1"]) + "\n" + str(connection["entity2"])
        rel_txt = str(connection["relationships"])

        prompt = EXPLAIN_PROMPT.format(
            ENTITIES=ent_txt,
            RELATIONS=rel_txt,
            QUESTION=question,
        )
        return self.llm.call(prompt)

    # ---------------------------
    # MAIN ENTRYPOINT
    # ---------------------------
    def ask(self, question: str):
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
        explanation = self.explain(question, connection)

        return {
            "answer": explanation,
            "mentions": mentions,
            "resolved": resolved,
            "relationships": connection["relationships"],
            "raw": connection["relationships"],
            "cypher": connection["cypher"],
            "params": connection["params"]
        }

