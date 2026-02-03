// Query 1: Find suppliers for Control Chip with shipping to factories in the same country.
// params: {"product_name": "Control Chip", "max_lead_time": "20"}
MATCH (p:Product {name: $product_name})-[:USES]->(c:Component)<-[:SUPPLIED_BY]-(s:Supplier)-[:LOCATED_IN]->(country:Country) MATCH (f:Factory)-[:LOCATED_IN]->(country) MATCH (s)-[ship:SHIPS_TO]->(f) WHERE ship.time_days <= $max_lead_time RETURN s.name AS Supplier, f.name AS Factory, ship.cost_usd AS ShippingCost ORDER BY ship.cost_usd LIMIT 50
// substituted
MATCH (p:Product {name: "Control Chip"})-[:USES]->(c:Component)<-[:SUPPLIED_BY]-(s:Supplier)-[:LOCATED_IN]->(country:Country) MATCH (f:Factory)-[:LOCATED_IN]->(country) MATCH (s)-[ship:SHIPS_TO]->(f) WHERE ship.time_days <= "20" RETURN s.name AS Supplier, f.name AS Factory, ship.cost_usd AS ShippingCost ORDER BY ship.cost_usd LIMIT 50
;

// Query 2: Count suppliers to ensure there are options available.
// params: {"product_name": "Control Chip", "max_lead_time": "20"}
MATCH (p:Product {name: $product_name})-[:USES]->(c:Component)<-[:SUPPLIED_BY]-(s:Supplier)-[:LOCATED_IN]->(country:Country) MATCH (f:Factory)-[:LOCATED_IN]->(country) MATCH (s)-[ship:SHIPS_TO]->(f) WHERE ship.time_days <= $max_lead_time RETURN COUNT(s) AS SupplierCount ORDER BY SupplierCount DESC LIMIT 1
// substituted
MATCH (p:Product {name: "Control Chip"})-[:USES]->(c:Component)<-[:SUPPLIED_BY]-(s:Supplier)-[:LOCATED_IN]->(country:Country) MATCH (f:Factory)-[:LOCATED_IN]->(country) MATCH (s)-[ship:SHIPS_TO]->(f) WHERE ship.time_days <= "20" RETURN COUNT(s) AS SupplierCount ORDER BY SupplierCount DESC LIMIT 1
;
