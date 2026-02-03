# modules/graph_store.py

import logging
from typing import List
from neo4j import GraphDatabase

from modules.schemas.graph_schema import Entity, Relation, GraphResult


class GraphStore:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self.log = logging.getLogger(__name__)

    def close(self):
        self.driver.close()

    # ---------------------------------------------------------
    # SCHEMA SUMMARY (unchanged, still useful)
    # ---------------------------------------------------------
    def get_schema_summary(self):
        with self.driver.session() as session:
            labels = session.run("CALL db.labels()").value()
            rel_types = session.run("CALL db.relationshipTypes()").value()
            props = session.run("""
                CALL db.schema.nodeTypeProperties()
                YIELD nodeLabels, propertyName
                RETURN nodeLabels, propertyName
            """).data()

        label_props = {}
        for row in props:
            for label in row["nodeLabels"]:
                label_props.setdefault(label, set()).add(row["propertyName"])

        return {
            "labels": labels,
            "relationships": rel_types,
            "properties": {k: sorted(list(v)) for k, v in label_props.items()},
        }

    # ---------------------------------------------------------
    # INGEST GRAPHRESULT INTO NEO4J
    # ---------------------------------------------------------
    def insert_graph(self, graph: GraphResult):
        """
        Insert a typed, semantic knowledge graph:
        - entity.type becomes actual Neo4j label
        - entity.attributes become node properties
        - aliases + source_spans stored on node
        - relationships are real typed edges
        """
        entities = [self._entity_to_params(e) for e in graph.entities]
        relations = [self._relation_to_params(r) for r in graph.relations]

        self.log.info(
            "Neo4j insert_graph: %d entities, %d relations",
            len(entities),
            len(relations),
        )

        with self.driver.session() as session:
            if entities:
                session.execute_write(self._upsert_entities, entities)
            if relations:
                session.execute_write(self._upsert_relations, relations)

    # ---------------------------------------------------------
    # PARAM CONVERSION
    # ---------------------------------------------------------
    @staticmethod
    def _entity_to_params(e: Entity) -> dict:
        # Flatten attributes list → property dict
        extra_props = {}
        for attr in e.attributes:
            if attr.key:
                extra_props[attr.key] = attr.value

        return {
            "id": e.id,
            "label": e.type.strip().replace(" ", "_"),  # Person, Company, Animal, etc.
            "name": e.name,
            "aliases": e.aliases or [],
            "source_spans": e.source_spans or [],
            "properties": extra_props,
        }

    @staticmethod
    def _relation_to_params(r: Relation) -> dict:
        return {
            "source_id": r.source_id,
            "target_id": r.target_id,
            "type": r.type.strip().upper().replace(" ", "_"),
            "source_span": r.source_span,
            "confidence": r.confidence,
        }

    # ---------------------------------------------------------
    # NEO4J WRITERS
    # ---------------------------------------------------------
    @staticmethod
    def _upsert_entities(tx, entities: List[dict]):
        for e in entities:
            label = e["label"] or "Thing"  # fallback
            tx.run(
                f"""
                MERGE (n:{label} {{id: $id}})
                SET n.name = $name,
                    n.aliases = $aliases,
                    n.source_spans = $source_spans
                SET n += $properties    // adds founded: 1998 etc.
                """,
                {
                    "id": e["id"],
                    "name": e["name"],
                    "aliases": e["aliases"],
                    "source_spans": e["source_spans"],
                    "properties": e["properties"],
                },
            )

    @staticmethod
    def _upsert_relations(tx, relations: List[dict]):
        for r in relations:
            rel_type = r["type"]
            tx.run(
                f"""
                MATCH (s {{id: $sid}})
                MATCH (t {{id: $tid}})
                MERGE (s)-[rel:{rel_type}]->(t)
                SET rel.source_span = $source_span,
                    rel.confidence = $confidence
                """,
                {
                    "sid": r["source_id"],
                    "tid": r["target_id"],
                    "source_span": r["source_span"],
                    "confidence": r["confidence"],
                },
            )

    # ---------------------------------------------------------
    # ENTITY RESOLUTION HELPERS
    # ---------------------------------------------------------

    def resolve_entity(self, term: str):
        """
        Resolve a name/alias to an entity node.
        Returns dict with id, name, label, aliases, etc.
        """
        with self.driver.session() as session:
            # Exact name match
            res = session.run("""
                MATCH (n)
                WHERE n.name = $term
                RETURN n LIMIT 1
            """, term=term).single()

            if res:
                node = res["n"]
                return {
                    "id": node["id"],
                    "name": node.get("name"),
                    "aliases": node.get("aliases", []),
                    "label": list(node.labels)[0]
                }

            # Alias match
            res = session.run("""
                MATCH (n)
                WHERE $term IN n.aliases
                RETURN n LIMIT 1
            """, term=term).single()

            if res:
                node = res["n"]
                return {
                    "id": node["id"],
                    "name": node.get("name"),
                    "aliases": node.get("aliases", []),
                    "label": list(node.labels)[0]
                }

        return None


    def find_relationships(self, id1: str, id2: str):
        """
        Returns:
        {
            "cypher": <cypher string>,
            "params": { "id1": ..., "id2": ... },
            "rows": [ ... list of relationship records ... ]
        }
        """
        cypher = """
            MATCH (a {id:$id1})-[r]-(b {id:$id2})
            RETURN type(r) AS rel_type,
                r.source_span AS source_span,
                r.confidence AS confidence
        """

        params = {"id1": id1, "id2": id2}

        with self.driver.session() as session:
            rows = session.run(cypher, **params).data()

        return {
            "cypher": cypher.strip(),
            "params": params,
            "rows": rows or []
        }
