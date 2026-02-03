// Query 1: Retrieve products that use the component impacted by the asteroid
// params: {"component_name": "asteroid impact"}
MATCH (comp:Component)<-[:USES]-(p:Product) WHERE comp.name = $component_name RETURN p.id AS product_id, p.name AS product_name LIMIT 50
// substituted
MATCH (comp:Component)<-[:USES]-(p:Product) WHERE comp.name = "asteroid impact" RETURN p.id AS product_id, p.name AS product_name LIMIT 50
;

// Query 2: Find products supplied by suppliers located in Vietnam
// params: {"country_name": "Vietnam"}
MATCH (s:Supplier)-[:SUPPLIES]->(comp:Component)<-[:USES]-(p:Product) WHERE s.country = $country_name RETURN DISTINCT p.id AS product_id, p.name AS product_name LIMIT 50
// substituted
MATCH (s:Supplier)-[:SUPPLIES]->(comp:Component)<-[:USES]-(p:Product) WHERE s.country = "Vietnam" RETURN DISTINCT p.id AS product_id, p.name AS product_name LIMIT 50
;

// Query 3: Identify factories producing products that use the impacted component
// params: {"component_name": "asteroid impact"}
MATCH (f:Factory)-[:PRODUCES]->(p:Product)<-[:USES]-(comp:Component) WHERE comp.name = $component_name RETURN f.id AS factory_id, f.name AS factory_name, p.id AS product_id, p.name AS product_name LIMIT 50
// substituted
MATCH (f:Factory)-[:PRODUCES]->(p:Product)<-[:USES]-(comp:Component) WHERE comp.name = "asteroid impact" RETURN f.id AS factory_id, f.name AS factory_name, p.id AS product_id, p.name AS product_name LIMIT 50
;
