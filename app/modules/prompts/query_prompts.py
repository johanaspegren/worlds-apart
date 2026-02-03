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
    - Include LIMIT 50 unless the query is an aggregate.
    - Relationship directions are STRICT and MUST follow the ontology exactly.
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
    - SUPPLIED_BY: (comp:Component)-[:SUPPLIED_BY]->(s:Supplier)
    - PRODUCES:    (f:Factory)-[:PRODUCES]->(p:Product)
    - LOCATED_IN:  (x)-[:LOCATED_IN]->(c:Country)
    - SHIPS_TO:    (s:Supplier)-[:SHIPS_TO]->(f:Factory)
    - EXPORTS_VIA: (f:Factory)-[:EXPORTS_VIA]->(port:Port)
    - IMPORTS_TO:  (port:Port)-[:IMPORTS_TO]->(c:Country)

    Forbidden (WRONG) patterns:
    - (s:Supplier)-[:SUPPLIED_BY]->(:Component)
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
    Use ONLY the provided Cypher results to answer the question.
    If results are empty or insufficient, say so clearly.

    Schema / Ontology:
    {SCHEMA}

    Scenario:
    {SCENARIO}

    Question:
    {QUESTION}

    Cypher Results:
    {RESULTS}

    Answer:
"""
