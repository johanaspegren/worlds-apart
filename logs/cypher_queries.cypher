// Query 1: Identify suppliers in Vietnam that ship to factories in Poland.
// params: {"supplier_country": "Vietnam", "factory_country": "Poland"}
MATCH (s:Supplier)-[ship:SHIPS_TO]->(f:Factory) WHERE s.country = $supplier_country AND f.country = $factory_country RETURN s.id AS supplier_id, f.id AS factory_id, ship.cost_usd AS shipping_cost LIMIT 50
// substituted
MATCH (s:Supplier)-[ship:SHIPS_TO]->(f:Factory) WHERE s.country = "Vietnam" AND f.country = "Poland" RETURN s.id AS supplier_id, f.id AS factory_id, ship.cost_usd AS shipping_cost LIMIT 50
;
