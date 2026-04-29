// Query 1: Identify the most common medications prescribed for the specified diagnosis.
// params: {"diagnosis_id": "diagnosis"}
MATCH (v:Visit)-[:RESULTED_IN]->(d:Diagnosis {id: $diagnosis_id})<-[:RESULTED_IN]-(v2:Visit)-[:PRESCRIBED]->(m:Medication) RETURN m.name AS medication_name, COUNT(m) AS medication_count ORDER BY medication_count DESC LIMIT 50
// substituted
MATCH (v:Visit)-[:RESULTED_IN]->(d:Diagnosis {id: "diagnosis"})<-[:RESULTED_IN]-(v2:Visit)-[:PRESCRIBED]->(m:Medication) RETURN m.name AS medication_name, COUNT(m) AS medication_count ORDER BY medication_count DESC LIMIT 50
;
