// Query 1: Finding products that are supplied by suppliers located in Vietnam.
// params: {"country_name": "Vietnam"}
MATCH (p:Product)-[:USES]->(comp:Component)-[:SUPPLIED_BY]->(s:Supplier)-[:LOCATED_IN]->(c:Country {name: $country_name}) RETURN p.id AS product_id, p.name AS product_name LIMIT 50
// substituted
MATCH (p:Product)-[:USES]->(comp:Component)-[:SUPPLIED_BY]->(s:Supplier)-[:LOCATED_IN]->(c:Country {name: "Vietnam"}) RETURN p.id AS product_id, p.name AS product_name LIMIT 50
;
