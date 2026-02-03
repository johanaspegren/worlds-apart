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
    - Include LIMIT 50 unless the query is an aggregate.
    - Return 1-3 queries that together answer the question.
    - If unsure, still return the best-effort query.

    Schema / Ontology:
    {SCHEMA}

    Scenario:
    {SCENARIO}

    Question:
    {QUESTION}

    Return ONLY JSON with this shape:
    {{
      "queries": [
        {{
          "cypher": "...",
          "params": {{"key": "value"}},
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
