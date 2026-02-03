# modules/prompts/query_prompts.py

MENTION_PROMPT = """
    Extract entity-like terms from the user question.
    Return a JSON list of exact text spans.

    Rules:
    - Include proper names, organizations, animals, locations, products, etc.
    - NO hallucinations.
    - Do not infer new names.
    - Keep only the text as written.

    Example:
    "How is Mary connected to Lily?"
    → ["Mary", "Lily"]

    Return ONLY JSON.
"""

EXPLAIN_PROMPT = """
    You are explaining a relationship between resolved entities in a knowledge graph.

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
