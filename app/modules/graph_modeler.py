# modules/graph_modeler.py

from modules.llm_handler import LLMHandler
from modules.schemas.graph_schema import (
    EntityList,
    RelationList,
    GraphResult,
)

ENTITY_PROMPT = """
    You extract real-world entities from text and return a minimal, clean entity list.

    GOAL:
    Identify actual things that exist in the world, not just words.

    For each entity:
    - Choose a canonical name (e.g. "Mary", "Lamb", "Megacorp Inc.", "San Francisco").
    - Infer a type from context, such as:
    Person, Company, Animal, City, Country, Industry, Product, Event, Organization, Other.
    - Merge alternative references or names into aliases:
    - "the lamb", "little lamb", "Lily" → aliases for the same Animal
    - "the company", "Megacorp" → aliases for "Megacorp Inc."
    - Capture short source_spans: exact snippets where the entity appears.
    - If there are clear attributes like founded year, revenue, etc.,
    you MAY include them in `attributes` as:
    {"key": "founded", "value": 1998}

    DO NOT:
    - Create entities for pronouns ("she", "it").
    - Create entities for adjectives or generic phrases ("little", "global").
    - Create entities for pure numbers ("38 countries"); those should be attributes or ignored.

    Return ONLY valid JSON:
    {"entities": [ ... ]}

    Example:
    Text: "Mary had a little lamb. The lamb was named Lily."

    Entities:
    {
    "entities": [
        {
        "id": "E1",
        "name": "Mary",
        "type": "Person",
        "aliases": ["the girl"],
        "source_spans": ["Mary", "the girl"],
        "attributes": []
        },
        {
        "id": "E2",
        "name": "Lamb",
        "type": "Animal",
        "aliases": ["little lamb", "Lily", "the lamb"],
        "source_spans": ["little lamb", "Lamb", "Lily", "the lamb"],
        "attributes": []
        }
    ]
    }

    Text:
    {TEXT}
"""

RELATION_PROMPT = """
    You extract meaningful semantic relationships between entities.

    Entities:
    {ENTITY_LIST}

    Rules:
    - Use ONLY the entity IDs from the list above.
    - DO NOT create new entities.
    - Only extract real-world relations such as:
    OWNS, HEADQUARTERED_IN, LOCATED_IN, SERVES, OPERATES_IN,
    WORKS_FOR, FOUNDED, PRODUCES, ACQUIRED, PART_OF, RELATED_TO.
    - DO NOT create relations about names/aliases (no HAS_NAME, HAS_ALIAS, etc).
    - If a sentence is only about naming ("The lamb was named Lily"),
    that is already captured as an alias on the entity, not a relation.
    - For each relation, include:
    - source_id
    - target_id
    - type  (UPPERCASE, no spaces, e.g. HEADQUARTERED_IN, OWNS)
    - source_span (short text fragment)
    - confidence (0.0–1.0)

    Return ONLY valid JSON:
    {"relations": [ ... ]}

    Example:
    Text: "Mary had a little lamb."
    Entities:
    E1: Mary (Person)
    E2: Lamb (Animal)

    Relations:
    {
    "relations": [
        {
        "source_id": "E1",
        "type": "OWNS",
        "target_id": "E2",
        "source_span": "Mary had a little lamb",
        "confidence": 0.95
        }
    ]
    }

    Text:
    {TEXT}
"""


class GraphModeler:
    """
    Minimal 2-pass graph extractor:
      1. Extract entities (with types, aliases, attributes)
      2. Extract relations between those entities
      3. Return GraphResult for Neo4j ingestion
    """

    def __init__(self, llm: LLMHandler):
        self.llm = llm

    def build_graph(self, text: str) -> GraphResult:
        snippet = text[:8000]  # keep it safe for prompt size

        # ---- PASS 1: ENTITIES ----
        ent_prompt = ENTITY_PROMPT.replace("{TEXT}", snippet)
        entity_list: EntityList = self.llm.call_schema_prompt(ent_prompt, EntityList)
        entities = entity_list.entities

        # ---- FORMAT ENTITIES FOR RELATION PROMPT ----
        entity_lines = [f"{e.id}: {e.name} ({e.type})" for e in entities]
        formatted_entities = "\n".join(entity_lines)

        # ---- PASS 2: RELATIONS ----
        rel_prompt = RELATION_PROMPT.replace("{ENTITY_LIST}", formatted_entities).replace("{TEXT}", snippet)
        relation_list: RelationList = self.llm.call_schema_prompt(rel_prompt, RelationList)
        relations = relation_list.relations

        return GraphResult(entities=entities, relations=relations)
