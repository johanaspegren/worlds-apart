// Query 1: Identify shipping costs from suppliers in Vietnam to factories in Poland for the Control Chip.
// params: {"supplier_country": "Vietnam", "factory_country": "Poland", "component_id": "C1"}
MATCH (s:Supplier)-[ship:SHIPS_TO]->(f:Factory) WHERE s.country = $supplier_country AND f.country = $factory_country MATCH (s)-[:SUPPLIES]->(comp:Component {id: $component_id}) RETURN ship.cost_usd AS shipping_cost, f.id AS factory_id, s.id AS supplier_id ORDER BY shipping_cost LIMIT 50
// substituted
MATCH (s:Supplier)-[ship:SHIPS_TO]->(f:Factory) WHERE s.country = "Vietnam" AND f.country = "Poland" MATCH (s)-[:SUPPLIES]->(comp:Component {id: "C1"}) RETURN ship.cost_usd AS shipping_cost, f.id AS factory_id, s.id AS supplier_id ORDER BY shipping_cost LIMIT 50
;

// Query 2: Compute export and import costs for the factory in Poland to the destination country.
// params: {"factory_country": "Poland", "destination_country": "Vietnam"}
MATCH (f:Factory)-[export:EXPORTS_VIA]->(port:Port) WHERE f.country = $factory_country MATCH (port)-[import:IMPORTS_TO]->(c:Country) WHERE c.name = $destination_country RETURN export.cost_usd AS export_cost, import.cost_usd AS import_cost, f.id AS factory_id, port.id AS port_id ORDER BY (export.cost_usd + import.cost_usd) LIMIT 50
// substituted
MATCH (f:Factory)-[export:EXPORTS_VIA]->(port:Port) WHERE f.country = "Poland" MATCH (port)-[import:IMPORTS_TO]->(c:Country) WHERE c.name = "Vietnam" RETURN export.cost_usd AS export_cost, import.cost_usd AS import_cost, f.id AS factory_id, port.id AS port_id ORDER BY (export.cost_usd + import.cost_usd) LIMIT 50
;
