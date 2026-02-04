# modules/graph_store.py

import logging
from typing import Dict, List, Any
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship

from app.modules.schemas.graph_schema import Entity, Relation, GraphResult


class GraphStore:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self.log = logging.getLogger(__name__)
        self._schema_summary: Dict | None = None

    def close(self):
        self.driver.close()

    def has_data(self) -> bool:
        with self.driver.session() as session:
            count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        return count > 0

    def run_cypher(self, cypher: str, params: dict | None = None) -> List[Dict]:
        with self.driver.session() as session:
            rows = session.run(cypher, **(params or {})).data()
        return [self._serialize_row(row) for row in (rows or [])]

    def _serialize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._serialize_value(value) for key, value in row.items()}

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, Node):
            labels = list(value.labels) if value.labels else []
            return {
                "_type": "node",
                "id": value.get("id"),
                "labels": labels,
                "properties": dict(value),
            }
        if isinstance(value, Relationship):
            return {
                "_type": "relationship",
                "type": value.type,
                "source": value.start_node.get("id"),
                "target": value.end_node.get("id"),
                "properties": dict(value),
            }
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._serialize_value(item) for key, item in value.items()}
        return value

    def find_name_matches(self, term: str, limit: int = 5) -> List[Dict]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (n)
                WHERE n.name = $term
                RETURN labels(n) AS labels, n.name AS name, n.id AS id
                LIMIT $limit
                """,
                term=term,
                limit=limit,
            ).data()
        return rows or []

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def fetch_subgraph(self, ids: List[str]) -> Dict[str, List[Dict]]:
        if not ids:
            return {"nodes": [], "edges": []}
        unique_ids = list({str(value) for value in ids if value})
        if not unique_ids:
            return {"nodes": [], "edges": []}
        with self.driver.session() as session:
            node_rows = session.run(
                "MATCH (n) WHERE n.id IN $ids RETURN n",
                ids=unique_ids,
            ).data()
            edge_rows = session.run(
                """
                MATCH (a)-[r]-(b)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id AS source, b.id AS target, type(r) AS type, properties(r) AS properties
                """,
                ids=unique_ids,
            ).data()

        nodes = []
        for row in node_rows or []:
            node = row.get("n")
            if node is None:
                continue
            labels = list(node.labels) if hasattr(node, "labels") else []
            node_id = node.get("id") if hasattr(node, "get") else None
            if not node_id:
                continue
            nodes.append(
                {
                    "id": str(node_id),
                    "name": node.get("name") if hasattr(node, "get") else None,
                    "label": labels[0] if labels else "Entity",
                    "properties": dict(node) if hasattr(node, "items") else {},
                }
            )

        edges = []
        for row in edge_rows or []:
            source_id = row.get("source")
            target_id = row.get("target")
            if not source_id or not target_id:
                continue
            edges.append(
                {
                    "source": str(source_id),
                    "target": str(target_id),
                    "type": row.get("type") or "RELATED_TO",
                    "properties": row.get("properties") or {},
                }
            )

        return {"nodes": nodes, "edges": edges}

    def insert_supply_chain(self, rows: List[Dict]):
        self.log.info("Neo4j insert_supply_chain: %d rows", len(rows))
        with self.driver.session() as session:
            session.execute_write(self._upsert_supply_chain, rows)

    # ---------------------------------------------------------
    # SCHEMA SUMMARY (unchanged, still useful)
    # ---------------------------------------------------------
    def get_schema_summary(self):
        if self._schema_summary is not None:
            return self._schema_summary
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

        self._schema_summary = {
            "labels": labels,
            "relationships": rel_types,
            "properties": {k: sorted(list(v)) for k, v in label_props.items()},
        }
        return self._schema_summary

    @staticmethod
    def get_supply_chain_ontology_text() -> str:
        return (
            "Nodes:\n"
            "- Product {id, name}\n"
            "- Component {id, name}\n"
            "- Supplier {id, name, country}\n"
            "- Factory {id, name, country}\n"
            "- Port {id, name, country}\n"
            "- Country {name}\n\n"
            "Relationships:\n"
            "- (Product)-[:USES]->(Component)\n"
            "- (Supplier)-[:SUPPLIES]->(Component)\n"
            "- (Factory)-[:PRODUCES]->(Product)\n"
            "- (Supplier)-[:LOCATED_IN]->(Country)\n"
            "- (Factory)-[:LOCATED_IN]->(Country)\n"
            "- (Port)-[:LOCATED_IN]->(Country)\n"
            "- (Supplier)-[:SHIPS_TO {row_id, cost_usd, time_days, co2_kg}]->(Factory)\n"
            "- (Factory)-[:EXPORTS_VIA {row_id, cost_usd, time_days, co2_kg}]->(Port)\n"
            "- (Port)-[:IMPORTS_TO {row_id, cost_usd, time_days, co2_kg}]->(Country)\n\n"
            "IDs:\n"
            "- product_id, component_id, supplier_id, factory_id, port_id are node ids.\n"
        )

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

    @staticmethod
    def _upsert_supply_chain(tx, rows: List[Dict]):
        cypher = """
            MERGE (product:Product {id: $product_id})
            SET product.name = $product_name

            MERGE (component:Component {id: $component_id})
            SET component.name = $component_name

            MERGE (supplier:Supplier {id: $supplier_id})
            SET supplier.name = $supplier_name,
                supplier.country = $supplier_country

            MERGE (factory:Factory {id: $factory_id})
            SET factory.name = $factory_name,
                factory.country = $factory_country

            MERGE (port:Port {id: $port_id})
            SET port.name = $port_name,
                port.country = $port_country

            MERGE (supplier_country:Country {name: $supplier_country})
            MERGE (factory_country:Country {name: $factory_country})
            MERGE (port_country:Country {name: $port_country})
            MERGE (market_country:Country {name: $market_country})

            MERGE (product)-[:USES {row_id: $row_id}]->(component)
            MERGE (component)-[:SUPPLIED_BY {row_id: $row_id}]->(supplier)
            MERGE (supplier)-[:SUPPLIES {row_id: $row_id}]->(component)
            MERGE (factory)-[:PRODUCES {row_id: $row_id}]->(product)

            MERGE (supplier)-[:LOCATED_IN]->(supplier_country)
            MERGE (factory)-[:LOCATED_IN]->(factory_country)
            MERGE (port)-[:LOCATED_IN]->(port_country)

            MERGE (supplier)-[ship:SHIPS_TO {row_id: $row_id}]->(factory)
            SET ship.cost_usd = $ship_cost_usd,
                ship.time_days = $ship_time_days,
                ship.co2_kg = $ship_co2_kg

            MERGE (factory)-[export:EXPORTS_VIA {row_id: $row_id}]->(port)
            SET export.cost_usd = $export_cost_usd,
                export.time_days = $export_time_days,
                export.co2_kg = $export_co2_kg

            MERGE (port)-[import:IMPORTS_TO {row_id: $row_id}]->(market_country)
            SET import.cost_usd = $import_cost_usd,
                import.time_days = $import_time_days,
                import.co2_kg = $import_co2_kg
        """
        for row in rows:
            tx.run(cypher, **row)

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
                properties(r) AS rel_props
        """

        params = {"id1": id1, "id2": id2}

        with self.driver.session() as session:
            rows = session.run(cypher, **params).data()

        return {
            "cypher": cypher.strip(),
            "params": params,
            "rows": rows or []
        }
