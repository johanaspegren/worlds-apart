// Query 1: Retrieve recent transactions initiated by accounts to identify any unusual patterns.
MATCH (a:Account)-[:INITIATED]->(t:Transaction) RETURN a.id AS account_id, t.id AS transaction_id, t.amount AS transaction_amount ORDER BY t.timestamp DESC LIMIT 50
;

// Query 2: Identify accounts using high-risk devices to assess potential fraud.
MATCH (a:Account)-[:USES_DEVICE]->(d:Device) RETURN a.id AS account_id, d.id AS device_id, d.type AS device_type ORDER BY a.risk_score DESC LIMIT 50
;

// Query 3: Check accounts logging in from suspicious IP addresses to detect potential fraud.
MATCH (a:Account)-[:LOGGED_IN_FROM]->(ip:IP) RETURN a.id AS account_id, ip.address AS ip_address ORDER BY a.risk_score DESC LIMIT 50
;
