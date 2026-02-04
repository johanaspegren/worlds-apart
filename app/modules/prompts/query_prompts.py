# modules/prompts/query_prompts.py

MENTION_PROMPT = """
    Extract entity-like terms from the user question.
    Use the schema/ontology below to prioritize terms that map to node types or properties.
    Return a JSON list of exact text spans.

    Rules:
    - Include proper names, organizations, animals, locations, products, etc.
    - NO hallucinations.
    - Do not infer new names.
    - Keep only the text as written.

    Example:
    "How is Mary connected to Lily?"
    → ["Mary", "Lily"]

    Schema / Ontology:
    {SCHEMA}

    Return ONLY JSON.
"""

EXPLAIN_PROMPT = """
    You are explaining a relationship between resolved entities in a knowledge graph.

    Schema / Ontology:
    {SCHEMA}

    Entities:
    {ENTITIES}

    Relationships:
    {RELATIONS}

    Scenario:
    {SCENARIO}

    Rules:
    - Mention which canonical entities were matched.
    - Mention if aliases were used.
    - Explain the relationship clearly and briefly.
    - If multiple relations exist, summarize them.

    User Question:
    {QUESTION}

    Answer:
"""

CYPHER_PROMPT = """
    You are a Neo4j Cypher generator for a supply chain knowledge graph.
    Use the schema/ontology below to write READ-ONLY queries.

    Rules:
    - Use only MATCH/RETURN/WHERE/WITH/ORDER BY/LIMIT.
    - Use parameters for literals (e.g., $product_name).
    - To access relationship properties, bind a relationship variable and read it:
      MATCH (s)-[ship:SHIPS_TO]->(f) RETURN ship.cost_usd
    - Do NOT put variable property values inside pattern maps like {{cost_usd: cost}}.
      Use WHERE and relationship variables instead.
    - If you reference ship/export/import variables, they MUST be declared in the MATCH:
      MATCH (s)-[ship:SHIPS_TO]->(f) ... RETURN ship.cost_usd AS shipping_cost
    - Resolved Entities are authoritative. If status=resolved, you MUST use the given label and id.
      Use node ids instead of names whenever an id is provided.
    - If status=ambiguous, choose the label that fits the canonical path.
    - If status=unresolved, do NOT invent a node type. Use other constraints instead.
    - Return 1-3 queries that together answer the question.
    - If the question asks for "cheapest" or "best way", prefer 2-3 queries:
      1) identify candidate paths/nodes, 2) compute costs, 3) list top results.
    - Include LIMIT 50 unless the query is an aggregate.
    - Relationship directions are STRICT and MUST follow the ontology exactly.
    - Prefer SUPPLIES when connecting Supplier -> Component unless the question explicitly asks for "supplied by".
    - Never reverse relationship directions.
    - Prefer existence and dependency paths over metrics unless explicitly requested.
    - Do NOT access relationship properties unless the question asks for cost, time, emissions, or optimisation.
    - Risk questions should focus on dependency, concentration, and single points of failure.

    Schema / Ontology:
    {SCHEMA}

    Live Schema Summary (from database):
    {SCHEMA_SUMMARY}

    Resolved Entities (AUTHORITATIVE):
    {RESOLVED_ENTITIES}

    Canonical direction patterns (COPY EXACTLY):
    - USES:        (p:Product)-[:USES]->(comp:Component)
    - SUPPLIES:    (s:Supplier)-[:SUPPLIES]->(comp:Component)
    - PRODUCES:    (f:Factory)-[:PRODUCES]->(p:Product)
    - LOCATED_IN:  (x)-[:LOCATED_IN]->(c:Country)
    - SHIPS_TO:    (s:Supplier)-[:SHIPS_TO]->(f:Factory)
    - EXPORTS_VIA: (f:Factory)-[:EXPORTS_VIA]->(port:Port)
    - IMPORTS_TO:  (port:Port)-[:IMPORTS_TO]->(c:Country)

    Forbidden (WRONG) patterns:
    - (:Component)-[:SUPPLIES]->(:Supplier)
    - (:Product)-[:PRODUCES]->(:Factory)
    - (:Country)-[:LOCATED_IN]->(:Supplier)

    Relationship names do NOT imply direction.
    Always follow the canonical direction patterns, even if they feel semantically reversed.

    Scenario:
    {SCENARIO}

    Question:
    {QUESTION}

    Return ONLY JSON with this shape:
    {{
      "queries": [
        {{
          "cypher": "SINGLE STRING. Do NOT split across keys.",
          "params": [{{"key": "param_name", "value": "param_value (string)"}}],
          "reason": "short reason"
        }}
      ]
    }}
"""

ANSWER_PROMPT = """
You are a supply chain graph assistant.

CRITICAL RULES:
- Use ONLY the provided Cypher results.
- Do NOT assume facts not present in the results.
- Every factual claim MUST be supported by the results.
- If results are empty or insufficient, say so clearly.
- Do NOT introduce new entities, paths, or metrics.

How to answer:
1. Start with a concise direct answer to the question.
2. Explain the reasoning using relationships and entities found in the results.
3. Refer explicitly to what the data shows (e.g. dependencies, costs, locations).
4. If multiple queries were used, combine them coherently.
5. If some expected paths were checked but returned no rows, state that.
6. Do NOT generalise beyond the scope of the results.

Schema / Ontology:
{SCHEMA}

Scenario:
{SCENARIO}

Question:
{QUESTION}

Cypher Results (AUTHORITATIVE EVIDENCE):
{RESULTS}

Answer (grounded, structured, and data-driven):

"""

VERIFY_PROMPT = """
You generate a single Cypher query to VERIFY the answer using the graph.

Rules:
- Use ONLY READ-ONLY clauses: MATCH/RETURN/WHERE/WITH/ORDER BY/LIMIT.
- Use parameters for literals (e.g., $product_id).
- The query must directly verify the key claims in the answer.
- Reuse entities and constraints already present in the executed queries/results.
- Prefer returning nodes/relationships referenced by the answer.
- Include LIMIT 50 unless it is an aggregate.
- Relationship directions are STRICT and MUST follow the ontology exactly.
- If verification is impossible from the data, return an empty cypher string.

GRAPH VISUALISATION RULES (CRITICAL):
- Prefer returning bound NODES and RELATIONSHIPS directly (e.g. RETURN p, s, comp).
- Do NOT project scalar properties (e.g. p.id, s.name) unless aggregation is required.
- The query is used to render a UI graph; returned variables should form a connected subgraph.

CRITICAL GRAPH RETURN RULE:
- You MUST return ALL node variables that appear in the MATCH clause.
- Do NOT return only properties or IDs if nodes are matched.
- The RETURN clause must expose the full evidence subgraph for UI rendering.
- Returning nodes is REQUIRED unless the query is purely an aggregate.

EXAMPLE (CORRECT):

MATCH (a)-[r:RELATES_TO]->(b)-[:DOES]->(c:C) WHERE a.id IN ["X1", "X2"] RETURN a, r, b, c
LIMIT 50


EXAMPLE (WRONG — DO NOT DO THIS):

MATCH (a)-[r:RELATES_TO]->(b)-[:DOES]->(c:C) WHERE a.id IN ["X1", "X2"] RETURN a, c
LIMIT 50



Schema / Ontology:
{SCHEMA}

Scenario:
{SCENARIO}

Question:
{QUESTION}

Answer to verify:
{ANSWER}

Executed queries:
{QUERIES}

Cypher results (evidence):
{RESULTS}

Return ONLY JSON with this shape:
{{
  "cypher": "SINGLE STRING. Do NOT split across keys.",
  "params": [{{"key": "param_name", "value": "param_value (string)"}}],
  "reason": "short reason"
}}
"""
