// Query 1: Identify suppliers of Product Gamma to understand potential disruptions.
// params: {"product_id": "P3"}
MATCH (s:Supplier)-[:SUPPLIES]->(comp:Component)<-[:USES]-(p:Product) WHERE p.id = $product_id RETURN s.id AS supplier_id, s.name AS supplier_name, s.country AS supplier_country LIMIT 50
// substituted
MATCH (s:Supplier)-[:SUPPLIES]->(comp:Component)<-[:USES]-(p:Product) WHERE p.id = "P3" RETURN s.id AS supplier_id, s.name AS supplier_name, s.country AS supplier_country LIMIT 50
;

// Query 2: Identify suppliers of Product Delta to compare with Product Gamma.
// params: {"product_id": "P4"}
MATCH (s:Supplier)-[:SUPPLIES]->(comp:Component)<-[:USES]-(p:Product) WHERE p.id = $product_id RETURN s.id AS supplier_id, s.name AS supplier_name, s.country AS supplier_country LIMIT 50
// substituted
MATCH (s:Supplier)-[:SUPPLIES]->(comp:Component)<-[:USES]-(p:Product) WHERE p.id = "P4" RETURN s.id AS supplier_id, s.name AS supplier_name, s.country AS supplier_country LIMIT 50
;

// Query 3: Find factories producing Product Gamma to assess impact of disruptions.
// params: {"product_id": "P3"}
MATCH (s:Supplier)-[:SHIPS_TO]->(f:Factory)-[:PRODUCES]->(p:Product) WHERE p.id = $product_id RETURN f.id AS factory_id, f.name AS factory_name, f.country AS factory_country LIMIT 50
// substituted
MATCH (s:Supplier)-[:SHIPS_TO]->(f:Factory)-[:PRODUCES]->(p:Product) WHERE p.id = "P3" RETURN f.id AS factory_id, f.name AS factory_name, f.country AS factory_country LIMIT 50
;
