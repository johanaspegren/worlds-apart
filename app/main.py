import csv
import io
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openpyxl import load_workbook

from app.modules.graph_store import GraphStore
from app.modules.llm_handler import LLMHandler
from app.modules.query_agent import QueryAgent
from app.modules.vector_store import SimpleVectorStore

from app.modules.file_utils import log_json, log_text


load_dotenv()

REQUIRED_COLUMNS = [
    "product_id",
    "product_name",
    "component_id",
    "component_name",
    "supplier_id",
    "supplier_name",
    "supplier_country",
    "factory_id",
    "factory_name",
    "factory_country",
    "port_id",
    "port_name",
    "port_country",
    "market_country",
    "ship_cost_usd",
    "ship_time_days",
    "ship_co2_kg",
    "export_cost_usd",
    "export_time_days",
    "export_co2_kg",
    "import_cost_usd",
    "import_time_days",
    "import_co2_kg",
]

NUMERIC_COLUMNS = {
    "ship_cost_usd": float,
    "ship_time_days": int,
    "ship_co2_kg": float,
    "export_cost_usd": float,
    "export_time_days": int,
    "export_co2_kg": float,
    "import_cost_usd": float,
    "import_time_days": int,
    "import_co2_kg": float,
}

NOTES_DIR = "notes"
DEV_PERSIST_DB_ENV = "DEV_PERSIST_DB"


@dataclass
class LLMConfig:
    provider: str
    model: str
    embed_model: str


@dataclass
class WorldState:
    rows: List[Dict] = field(default_factory=list)
    notes: Dict[int, str] = field(default_factory=dict)
    vector_store: SimpleVectorStore = field(default_factory=SimpleVectorStore)
    last_scenario: Dict = field(default_factory=dict)
    llm_config: LLMConfig | None = None
    graph_store: GraphStore | None = None
    def reset(self) -> None:
        self.rows = []
        self.notes = {}
        self.vector_store = SimpleVectorStore()
        self.last_scenario = {}


STATE = WorldState()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def dev_persist_enabled() -> bool:
    return is_truthy(os.getenv(DEV_PERSIST_DB_ENV))


def load_notes_from_disk() -> None:
    if STATE.notes:
        return
    if not os.path.isdir(NOTES_DIR):
        return
    notes: Dict[int, str] = {}
    for filename in sorted(os.listdir(NOTES_DIR)):
        if not filename.startswith("note_row_") or not filename.endswith(".txt"):
            continue
        try:
            note_id = int(filename.replace("note_row_", "").replace(".txt", ""))
        except ValueError:
            continue
        path = os.path.join(NOTES_DIR, filename)
        with open(path, "r", encoding="utf-8") as handle:
            notes[note_id] = handle.read()
    STATE.notes = notes


def ensure_vector_store(llm: LLMHandler) -> None:
    if STATE.vector_store.docs:
        return
    if not STATE.notes:
        load_notes_from_disk()
    if STATE.notes:
        STATE.vector_store = build_vector_store(llm)


@app.get("/")
def index() -> HTMLResponse:
    return FileResponse("static/index.html")


@app.post("/data/upload")
def upload_data(
    file: UploadFile = File(...),
    provider: str | None = Form(default=None),
    model: str | None = Form(default=None),
    embed_model: str | None = Form(default=None),
) -> Dict:
    STATE.reset()
    filename = file.filename or ""
    content = file.file.read()
    try:
        if filename.lower().endswith(".csv"):
            rows = parse_csv(content)
        elif filename.lower().endswith(".xlsx"):
            rows = parse_xlsx(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    STATE.rows = rows
    build_notes(rows)

    llm_config = resolve_llm_config(provider, model, embed_model)
    STATE.llm_config = llm_config

    llm = build_llm(llm_config)
    try:
        STATE.vector_store = build_vector_store(llm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding ingestion failed: {exc}") from exc

    try:
        graph_store = get_graph_store()
        graph_store.clear_graph()
        graph_store.insert_supply_chain(rows)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Neo4j ingestion failed: {exc}") from exc

    STATE.last_scenario = {}

    return {
        "status": "ok",
        "rows_loaded": len(rows),
        "schema_valid": True,
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.model,
        "embed_model": llm_config.embed_model,
    }


@app.post("/chat/rag")
def chat_rag(payload: Dict) -> Dict:
    print("RAG PAYLOAD:\n", payload)
    question = (payload.get("question") or "").strip()
    scenario = payload.get("scenario") or {}
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    llm_config = resolve_llm_config(
        payload.get("provider"),
        payload.get("model"),
        payload.get("embed_model"),
    )
    llm = build_llm(llm_config)

    if not STATE.rows and not dev_persist_enabled():
        raise HTTPException(status_code=400, detail="No data uploaded")

    ensure_vector_store(llm)
    retrieved = retrieve_notes(question, llm, STATE.vector_store)
    print("RAG RETRIEVED NOTES:\n", retrieved)
    log_json("rag_retrieved.json", {"retrieved": retrieved})
    scenario_text = scenario_summary(scenario)
    print("RAG SCENARIO TEXT:\n", scenario_text)
    log_json("rag_scenario.json", {"scenario_text": scenario_text})
    table_context = ""
    if STATE.rows and len(STATE.rows) <= 200:
        table_context = rows_to_csv(STATE.rows, max_rows=200)
    answer = rag_answer(llm, question, scenario_text, retrieved, table_context or None)
    print("RAG ANSWER:\n", answer)
    log_json("rag_answer.json", {"answer": answer})
    retval = {
        "answer": answer,
        "notes": retrieved,
        "retrieval": {
            "top_k": len(retrieved),
            "matches": retrieved,
        },
        "scenario": scenario_text,
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.model,
    }
    log_json("rag_response.json", retval)
    return retval


@app.post("/chat/graphrag")
def chat_graphrag(payload: Dict) -> Dict:
    print("GRAPHRAG PAYLOAD:\n", payload)
    question = (payload.get("question") or "").strip()
    scenario = payload.get("scenario") or {}
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    llm_config = resolve_llm_config(
        payload.get("provider"),
        payload.get("model"),
        payload.get("embed_model"),
    )
    llm = build_llm(llm_config)

    scenario_text = scenario_summary(scenario)
    graph_store = get_graph_store()
    agent = QueryAgent(llm, graph_store)

    if not STATE.rows and not dev_persist_enabled():
        raise HTTPException(status_code=400, detail="No data uploaded")
    if not STATE.rows and dev_persist_enabled() and not graph_store.has_data():
        raise HTTPException(status_code=400, detail="No graph data available")

    connection = agent.ask_cypher(question, scenario_text)

    retval = {
        "answer": connection.get("answer"),
        "scenario": scenario_text,
        "queries": connection.get("queries"),
        "results": connection.get("results"),
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.model,
    }
    print("GRAPHRAG RESPONSE:\n", retval)
    log_json("graphrag_response.json", retval)
    return retval


@app.post("/chat/rag/stream")
def chat_rag_stream(payload: Dict):
    question = (payload.get("question") or "").strip()
    scenario = payload.get("scenario") or {}
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    llm_config = resolve_llm_config(
        payload.get("provider"),
        payload.get("model"),
        payload.get("embed_model"),
    )
    llm = build_llm(llm_config)

    if not STATE.rows and not dev_persist_enabled():
        raise HTTPException(status_code=400, detail="No data uploaded")

    ensure_vector_store(llm)
    retrieved = retrieve_notes(question, llm, STATE.vector_store)
    scenario_text = scenario_summary(scenario)
    table_context = ""
    if STATE.rows and len(STATE.rows) <= 200:
        table_context = rows_to_csv(STATE.rows, max_rows=200)

    prompt = build_rag_prompt(question, scenario_text, retrieved, table_context or None)

    def event_stream():
        yield f"data: {json.dumps({'type': 'status', 'message': 'Retrieving context...'})}\n\n"
        yield f"data: {json.dumps({'type': 'meta', 'retrieval': retrieved, 'scenario': scenario_text})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"
        for chunk in llm.stream(prompt, temperature=0.2):
            if "error" in chunk:
                yield f"data: {json.dumps({'type': 'error', 'message': chunk['error']})}\n\n"
                return
            content = chunk.get("content", "")
            if content:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/graphrag/stream")
def chat_graphrag_stream(payload: Dict):
    question = (payload.get("question") or "").strip()
    scenario = payload.get("scenario") or {}
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    llm_config = resolve_llm_config(
        payload.get("provider"),
        payload.get("model"),
        payload.get("embed_model"),
    )
    llm = build_llm(llm_config)

    if not STATE.rows and not dev_persist_enabled():
        raise HTTPException(status_code=400, detail="No data uploaded")

    scenario_text = scenario_summary(scenario)
    graph_store = get_graph_store()
    agent = QueryAgent(llm, graph_store)
    if not STATE.rows and dev_persist_enabled() and not graph_store.has_data():
        raise HTTPException(status_code=400, detail="No graph data available")

    def event_stream():
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating Cypher queries...'})}\n\n"
        queries = agent.generate_cypher_queries_only(question, scenario_text)
        if queries:
            log_json("cypher_queries.json", {"queries": queries})
        yield f"data: {json.dumps({'type': 'status', 'message': 'Querying database...'})}\n\n"
        results = agent.execute_cypher_queries(queries) if queries else []
        log_json("cypher_execution_results.json", {"results": results})
        payload = {
            "type": "queries",
            "queries": queries,
            "results": results,
            "scenario": scenario_text,
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"
        prompt = agent.build_cypher_answer_prompt(question, scenario_text, results)
        for chunk in llm.stream(prompt, temperature=0.2):
            if "error" in chunk:
                yield f"data: {json.dumps({'type': 'error', 'message': chunk['error']})}\n\n"
                return
            content = chunk.get("content", "")
            if content:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def chat_graphrag_legacy(payload: Dict) -> Dict:
    question = (payload.get("question") or "").strip()
    scenario = payload.get("scenario") or {}
    if not STATE.rows:
        raise HTTPException(status_code=400, detail="No data uploaded")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    llm_config = resolve_llm_config(
        payload.get("provider"),
        payload.get("model"),
        payload.get("embed_model"),
    )
    llm = build_llm(llm_config)

    scenario_text = scenario_summary(scenario)
    graph_store = get_graph_store()
    agent = QueryAgent(llm, graph_store)
    connection = agent.ask(question, scenario_text)

    relationships = connection.get("relationships", [])
    trace = build_relationship_trace(relationships)
    trace_summary = f"Found {len(relationships)} graph relationships."

    if not relationships:
        schema_summary = graph_store.get_schema_summary()
        retrieved = retrieve_notes(question, llm, STATE.vector_store)
        answer = graphrag_fallback_answer(
            llm,
            question,
            scenario_text,
            schema_summary,
            retrieved,
        )
        connection["answer"] = answer
        connection["retrieval"] = {
            "top_k": len(retrieved),
            "matches": retrieved,
        }
        trace_summary = "No explicit relationships found; answered with schema + retrieved notes."

    return {
        "answer": connection.get("answer"),
        "trace": trace,
        "trace_summary": trace_summary,
        "scenario": scenario_text,
        "cypher": connection.get("cypher"),
        "params": connection.get("params"),
        "retrieval": connection.get("retrieval"),
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.model,
    }


@app.post("/chat/both")
def chat_both(payload: Dict) -> Dict:
    rag = chat_rag(payload)
    graphrag = chat_graphrag(payload)
    return {"rag": rag, "graphrag": graphrag}


def parse_csv(content: bytes) -> List[Dict]:
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return validate_rows(list(reader))


def parse_xlsx(content: bytes) -> List[Dict]:
    workbook = load_workbook(io.BytesIO(content), data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError("No rows found in workbook")
    headers = [str(cell).strip() if cell is not None else "" for cell in rows[0]]
    data_rows = []
    for row in rows[1:]:
        data = {headers[i]: row[i] for i in range(len(headers))}
        data_rows.append(data)
    return validate_rows(data_rows)


def validate_rows(raw_rows: List[Dict]) -> List[Dict]:
    if not raw_rows:
        raise ValueError("No data rows found")
    columns = list(raw_rows[0].keys())
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    validated = []
    for index, row in enumerate(raw_rows, start=2):
        validated_row = {}
        for col in REQUIRED_COLUMNS:
            value = row.get(col)
            if value is None or str(value).strip() == "":
                raise ValueError(f"Row {index}: {col} is required")
            if col in NUMERIC_COLUMNS:
                try:
                    cast_value = NUMERIC_COLUMNS[col](value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Row {index}: {col} must be numeric") from exc
                validated_row[col] = cast_value
            else:
                validated_row[col] = str(value).strip()
        validated_row["row_id"] = index
        validated.append(validated_row)
    return validated


def build_notes(rows: List[Dict]) -> None:
    os.makedirs(NOTES_DIR, exist_ok=True)
    notes = {}
    for row in rows:
        note = (
            f"For {row['product_name']}, the component {row['component_name']} is supplied by "
            f"{row['supplier_name']} in {row['supplier_country']}.\n"
            f"It is shipped to {row['factory_name']} at a cost of ${row['ship_cost_usd']}, "
            f"taking {row['ship_time_days']} days and emitting {row['ship_co2_kg']} kg CO₂.\n"
            f"The factory exports via {row['port_name']}, costing ${row['export_cost_usd']}, "
            f"taking {row['export_time_days']} days and emitting {row['export_co2_kg']} kg CO₂.\n"
            f"The shipment is imported to {row['market_country']} with cost "
            f"${row['import_cost_usd']}, time {row['import_time_days']} days and emissions "
            f"{row['import_co2_kg']} kg CO₂."
        )
        notes[row["row_id"]] = note
        with open(os.path.join(NOTES_DIR, f"note_row_{row['row_id']}.txt"), "w", encoding="utf-8") as handle:
            handle.write(note)
    STATE.notes = notes


def rows_to_csv(rows: List[Dict], max_rows: int = 200) -> str:
    if not rows:
        return ""
    headers = [col for col in REQUIRED_COLUMNS if col in rows[0]]
    def render_cell(value: object) -> str:
        text = "" if value is None else str(value)
        if any(ch in text for ch in [",", "\"", "\n"]):
            text = text.replace("\"", "\"\"")
            return f"\"{text}\""
        return text
    lines = [",".join(headers)]
    for row in rows[:max_rows]:
        line = ",".join(render_cell(row.get(col)) for col in headers)
        lines.append(line)
    return "\n".join(lines)


def build_vector_store(llm: LLMHandler) -> SimpleVectorStore:
    store = SimpleVectorStore()
    for note_id, text in STATE.notes.items():
        embedding = llm.embed(text)
        store.add(str(note_id), text, embedding)
    return store


def retrieve_notes(
    question: str, llm: LLMHandler, store: SimpleVectorStore, top_k: int = 5
) -> List[Dict]:
    if not question or not store.docs:
        return []
    query_embedding = llm.embed(question)
    matches = store.query(query_embedding, n=top_k)
    return [
        {
            "id": int(match["id"]),
            "score": match["score"],
            "text": match["text"],
        }
        for match in matches
    ]


def scenario_summary(scenario: Dict) -> str:
    parts = []
    if scenario.get("supplierBOutage"):
        parts.append("Supplier B outage is active.")
    if scenario.get("carbonTaxEnabled"):
        rate = scenario.get("carbonTaxRate", 0.1)
        parts.append(f"Carbon tax is enabled at ${rate}/kg CO₂.")
    max_days = scenario.get("maxLeadTimeDays")
    if max_days:
        parts.append(f"Max lead time is {max_days} days.")
    if not parts:
        return "No scenario constraints applied."
    return " ".join(parts)


def rag_answer(
    llm: LLMHandler,
    question: str,
    scenario_text: str,
    retrieved: List[Dict],
    table_context: str | None = None,
) -> str:
    prompt = build_rag_prompt(question, scenario_text, retrieved, table_context)
    #print("RAG PROMPT:\n", prompt)
    log_text("rag_prompt.txt", prompt)
    return llm.call(prompt, temperature=0.2)


def build_rag_prompt(
    question: str,
    scenario_text: str,
    retrieved: List[Dict],
    table_context: str | None = None,
) -> str:
    notes_text = "\n\n".join([note["text"] for note in retrieved])
    table_block = f"Table (full data):\n{table_context}\n\n" if table_context else ""
    prompt = (
        "You are a supply chain analyst. Use ONLY the provided context to answer. "
        "If the context does not contain the answer, say you do not have enough evidence.\n\n"
        f"Scenario: {scenario_text}\n\n"
        f"Context:\n{notes_text}\n\n"
        f"{table_block}"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt


def graphrag_fallback_answer(
    llm: LLMHandler,
    question: str,
    scenario_text: str,
    schema_summary: Dict,
    retrieved: List[Dict],
) -> str:
    notes_text = "\n\n".join([note["text"] for note in retrieved])
    prompt = (
        "You are a supply chain graph assistant. Use the schema summary and retrieved notes "
        "to answer the question. If the answer is not present, explain the limitation.\n\n"
        f"Scenario: {scenario_text}\n\n"
        f"Graph Schema: {schema_summary}\n\n"
        f"Retrieved Notes:\n{notes_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    print("GRAPHRAG FALLBACK PROMPT:\n", prompt)
    return llm.call(prompt, temperature=0.2)


def build_relationship_trace(relationships: List[Dict]) -> List[Dict]:
    trace = []
    for rel in relationships:
        rel_type = rel.get("rel_type", "REL")
        rel_props = rel.get("rel_props") or {}
        confidence = rel_props.get("confidence")
        source_span = rel_props.get("source_span")
        summary_parts = [rel_type]
        if confidence is not None:
            summary_parts.append(f"confidence {confidence:.2f}")
        if source_span:
            summary_parts.append(f"span: {source_span}")
        trace.append(
            {
                "rel_type": rel_type,
                "confidence": confidence,
                "source_span": source_span,
                "rel_props": rel_props,
                "summary": " | ".join(summary_parts),
            }
        )
    print("RELATIONSHIP TRACE:\n", trace)
    return trace


def resolve_llm_config(
    provider: str | None,
    model: str | None,
    embed_model: str | None,
) -> LLMConfig:
    resolved_provider = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    resolved_model = (model or os.getenv("LLM_MODEL") or default_model(resolved_provider)).strip()
    resolved_embed = (
        embed_model
        or os.getenv("LLM_EMBED_MODEL")
        or default_embed_model(resolved_provider)
    ).strip()
    return LLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        embed_model=resolved_embed,
    )


def default_model(provider: str) -> str:
    return {
        "openai": "gpt-4o-mini",
        "ollama": "llama3.1",
    }.get(provider, "gpt-4o-mini")


def default_embed_model(provider: str) -> str:
    return {
        "openai": "text-embedding-3-small",
        "ollama": "nomic-embed-text",
    }.get(provider, "text-embedding-3-small")


def build_llm(config: LLMConfig) -> LLMHandler:
    try:
        return LLMHandler(
            provider=config.provider,
            model=config.model,
            embed_model=config.embed_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM init failed: {exc}") from exc


def get_graph_store() -> GraphStore:
    if STATE.graph_store:
        return STATE.graph_store
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not password:
        raise HTTPException(status_code=500, detail="Missing Neo4j credentials in environment")
    try:
        store = GraphStore(uri, user, password)
        with store.driver.session() as session:
            session.run("RETURN 1")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Neo4j connection failed: {exc}") from exc
    STATE.graph_store = store
    return store
