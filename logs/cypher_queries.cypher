// Query 1: Find suppliers that supply the Control Chip used by Product Gamma.
MATCH (p:Product {id: 'P3'})-[:USES]->(comp:Component {id: 'C1'})<-[:SUPPLIES]-(s:Supplier) RETURN s.id AS supplier_id, s.name AS supplier_name
;
