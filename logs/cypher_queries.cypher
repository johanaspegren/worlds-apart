// Query 1: Find accounts that share the same device as Account A17.
// params: {"account_id": "A17"}
MATCH (a:Account)-[:USES_DEVICE]->(d:Device)<-[:USES_DEVICE]-(other:Account) WHERE a.id = $account_id RETURN other.id AS shared_account_id, other.name AS shared_account_name LIMIT 50
// substituted
MATCH (a:Account)-[:USES_DEVICE]->(d:Device)<-[:USES_DEVICE]-(other:Account) WHERE a.id = "A17" RETURN other.id AS shared_account_id, other.name AS shared_account_name LIMIT 50
;
