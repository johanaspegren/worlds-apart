import argparse
import json
import os
from typing import Any, Dict
from pprint import pprint

from app.main import (
    STATE,
    build_llm,
    ensure_vector_store,
    load_notes_from_disk,
    rag_answer,
    resolve_llm_config,
    retrieve_notes,
    scenario_summary,
)
from app.modules.file_utils import log_json



provider = "openai"
model = "gpt-4o-mini"
embed_model = "text-embedding-3-small"
question = "Why does a disruption in Vietnam affect Product Gamma but not Product Delta?"
top_k = 5

llm_config = resolve_llm_config(provider, model, embed_model)
llm = build_llm(llm_config)

ensure_vector_store(llm)

retrieved = retrieve_notes(question, llm, STATE.vector_store, top_k=top_k)
print("RAG RETRIEVED NOTES:\n", retrieved)
pprint(retrieved)

scenario_text = None

result: Dict[str, Any] = {
    "question": question,
    "scenario": scenario_text,
    "retrieved": retrieved,
    "retrieval": {
        "top_k": len(retrieved),
        "matches": retrieved,
    },
    "llm_provider": llm_config.provider,
    "llm_model": llm_config.model,
    "embed_model": llm_config.embed_model,
}

answer = rag_answer(llm, question, scenario_text, retrieved)
result["answer"] = answer
pprint(result['answer'])

print(json.dumps(result, indent=2, ensure_ascii=False))

