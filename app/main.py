import csv
import io
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from openpyxl import load_workbook

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


@dataclass
class GraphStore:
    nodes: Dict[str, set] = field(default_factory=lambda: {
        "product": set(),
        "component": set(),
        "supplier": set(),
        "factory": set(),
        "port": set(),
        "country": set(),
    })
    edges: List[Dict] = field(default_factory=list)


@dataclass
class VectorIndex:
    vocabulary: Dict[str, int] = field(default_factory=dict)
    idf: List[float] = field(default_factory=list)
    vectors: List[List[float]] = field(default_factory=list)
    note_ids: List[int] = field(default_factory=list)


@dataclass
class WorldState:
    rows: List[Dict] = field(default_factory=list)
    notes: Dict[int, str] = field(default_factory=dict)
    graph: GraphStore = field(default_factory=GraphStore)
    vector_index: VectorIndex = field(default_factory=VectorIndex)
    last_scenario: Dict = field(default_factory=dict)

    def reset(self) -> None:
        self.rows = []
        self.notes = {}
        self.graph = GraphStore()
        self.vector_index = VectorIndex()
        self.last_scenario = {}


STATE = WorldState()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> HTMLResponse:
    return FileResponse("static/index.html")


@app.post("/data/upload")
def upload_data(file: UploadFile = File(...)) -> Dict:
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
    build_graph_store(rows)
    build_notes(rows)
    build_vector_index()
    STATE.last_scenario = {}

    return {"status": "ok", "rows_loaded": len(rows), "schema_valid": True}


@app.post("/chat/rag")
def chat_rag(payload: Dict) -> Dict:
    question = (payload.get("question") or "").strip()
    scenario = payload.get("scenario") or {}
    if not STATE.rows:
        raise HTTPException(status_code=400, detail="No data uploaded")
    retrieved = retrieve_notes(question)
    scenario_text = scenario_summary(scenario)
    answer = rag_answer(question, scenario_text, retrieved)
    return {"answer": answer, "notes": retrieved, "scenario": scenario_text}


@app.post("/chat/graphrag")
def chat_graphrag(payload: Dict) -> Dict:
    question = (payload.get("question") or "").strip()
    scenario = payload.get("scenario") or {}
    if not STATE.rows:
        raise HTTPException(status_code=400, detail="No data uploaded")
    response = graphrag_answer(question, scenario)
    return response


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


def build_graph_store(rows: List[Dict]) -> None:
    graph = GraphStore()
    for row in rows:
        graph.nodes["product"].add(row["product_id"])
        graph.nodes["component"].add(row["component_id"])
        graph.nodes["supplier"].add(row["supplier_id"])
        graph.nodes["factory"].add(row["factory_id"])
        graph.nodes["port"].add(row["port_id"])
        graph.nodes["country"].add(row["supplier_country"])
        graph.nodes["country"].add(row["factory_country"])
        graph.nodes["country"].add(row["port_country"])
        graph.nodes["country"].add(row["market_country"])
        graph.edges.extend(
            [
                {
                    "from": ("product", row["product_id"]),
                    "to": ("component", row["component_id"]),
                    "type": "USES",
                    "row_id": row["row_id"],
                },
                {
                    "from": ("component", row["component_id"]),
                    "to": ("supplier", row["supplier_id"]),
                    "type": "SUPPLIED_BY",
                    "row_id": row["row_id"],
                },
                {
                    "from": ("supplier", row["supplier_id"]),
                    "to": ("country", row["supplier_country"]),
                    "type": "LOCATED_IN",
                    "row_id": row["row_id"],
                },
                {
                    "from": ("factory", row["factory_id"]),
                    "to": ("product", row["product_id"]),
                    "type": "PRODUCES",
                    "row_id": row["row_id"],
                },
                {
                    "from": ("supplier", row["supplier_id"]),
                    "to": ("factory", row["factory_id"]),
                    "type": "SHIPS_TO",
                    "row_id": row["row_id"],
                    "cost_usd": row["ship_cost_usd"],
                    "time_days": row["ship_time_days"],
                    "co2_kg": row["ship_co2_kg"],
                },
                {
                    "from": ("factory", row["factory_id"]),
                    "to": ("port", row["port_id"]),
                    "type": "EXPORTS_VIA",
                    "row_id": row["row_id"],
                    "cost_usd": row["export_cost_usd"],
                    "time_days": row["export_time_days"],
                    "co2_kg": row["export_co2_kg"],
                },
                {
                    "from": ("port", row["port_id"]),
                    "to": ("country", row["market_country"]),
                    "type": "IMPORTS_TO",
                    "row_id": row["row_id"],
                    "cost_usd": row["import_cost_usd"],
                    "time_days": row["import_time_days"],
                    "co2_kg": row["import_co2_kg"],
                },
            ]
        )
    STATE.graph = graph


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


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vector_index() -> None:
    docs = list(STATE.notes.items())
    vocabulary: Dict[str, int] = {}
    doc_tokens = []
    for _, text in docs:
        tokens = tokenize(text)
        doc_tokens.append(tokens)
        for token in set(tokens):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    doc_count = len(docs)
    idf = [0.0] * len(vocabulary)
    for token, idx in vocabulary.items():
        doc_freq = sum(1 for tokens in doc_tokens if token in tokens)
        idf[idx] = math.log((1 + doc_count) / (1 + doc_freq)) + 1
    vectors = []
    for tokens in doc_tokens:
        vec = [0.0] * len(vocabulary)
        for token in tokens:
            vec[vocabulary[token]] += 1
        norm = math.sqrt(sum((vec[i] * idf[i]) ** 2 for i in range(len(vec)))) or 1.0
        weighted = [(vec[i] * idf[i]) / norm for i in range(len(vec))]
        vectors.append(weighted)
    STATE.vector_index = VectorIndex(
        vocabulary=vocabulary,
        idf=idf,
        vectors=vectors,
        note_ids=[doc_id for doc_id, _ in docs],
    )


def retrieve_notes(question: str, top_k: int = 5) -> List[Tuple[int, str]]:
    index = STATE.vector_index
    if not index.vectors:
        return []
    query_tokens = tokenize(question)
    vec = [0.0] * len(index.vocabulary)
    for token in query_tokens:
        if token in index.vocabulary:
            vec[index.vocabulary[token]] += 1
    norm = math.sqrt(sum((vec[i] * index.idf[i]) ** 2 for i in range(len(vec)))) or 1.0
    query_vec = [(vec[i] * index.idf[i]) / norm for i in range(len(vec))]
    scored = []
    for idx, doc_vec in enumerate(index.vectors):
        score = sum(query_vec[i] * doc_vec[i] for i in range(len(query_vec)))
        scored.append((score, index.note_ids[idx]))
    scored.sort(reverse=True)
    results = []
    for _, note_id in scored[:top_k]:
        results.append((note_id, STATE.notes[note_id]))
    return results


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


def rag_answer(question: str, scenario_text: str, retrieved: List[Tuple[int, str]]) -> str:
    notes_text = "\n\n".join([note for _, note in retrieved])
    return (
        "Neural Recall (RAG) response based on retrieved notes.\n"
        f"Scenario: {scenario_text}\n"
        f"Question: {question}\n\n"
        "Retrieved evidence (top 5):\n"
        f"{notes_text}\n\n"
        "This response paraphrases retrieved rows and does not perform structured reasoning."
    )


def is_supplier_b(row: Dict) -> bool:
    supplier_id = row["supplier_id"].strip().lower()
    supplier_name = row["supplier_name"].strip().lower()
    return supplier_id in {"b", "supplier_b", "supplier b"} or "supplier b" in supplier_name


def calculate_path_metrics(row: Dict, scenario: Dict) -> Dict:
    base_cost = row["ship_cost_usd"] + row["export_cost_usd"] + row["import_cost_usd"]
    total_time = row["ship_time_days"] + row["export_time_days"] + row["import_time_days"]
    total_co2 = row["ship_co2_kg"] + row["export_co2_kg"] + row["import_co2_kg"]
    carbon_cost = 0.0
    if scenario.get("carbonTaxEnabled"):
        rate = float(scenario.get("carbonTaxRate", 0.1))
        carbon_cost = total_co2 * rate
    total_cost = base_cost + carbon_cost
    return {
        "base_cost": base_cost,
        "total_time": total_time,
        "total_co2": total_co2,
        "carbon_cost": carbon_cost,
        "total_cost": total_cost,
    }


def graphrag_answer(question: str, scenario: Dict) -> Dict:
    max_days = scenario.get("maxLeadTimeDays")
    filtered_rows = []
    exclusions = []
    for row in STATE.rows:
        if scenario.get("supplierBOutage") and is_supplier_b(row):
            exclusions.append((row, "Supplier B outage"))
            continue
        metrics = calculate_path_metrics(row, scenario)
        if max_days and metrics["total_time"] > max_days:
            exclusions.append((row, f"Lead time {metrics['total_time']} > {max_days}"))
            continue
        filtered_rows.append((row, metrics))

    response = {
        "answer": "",
        "trace": [],
        "scenario": scenario_summary(scenario),
    }

    question_lower = question.lower()
    if "which product" in question_lower and "pause" in question_lower:
        products = {row["product_id"]: row["product_name"] for row in STATE.rows}
        viable = {row["product_id"] for row, _ in filtered_rows}
        paused = [name for pid, name in products.items() if pid not in viable]
        if paused:
            response["answer"] = (
                "Pause these products because no feasible paths remain under the scenario constraints: "
                + ", ".join(paused)
                + "."
            )
        else:
            response["answer"] = "No products need to be paused; each has at least one feasible path."
    else:
        best_by_product: Dict[str, Tuple[Dict, Dict]] = {}
        for row, metrics in filtered_rows:
            current = best_by_product.get(row["product_id"])
            if current is None or metrics["total_cost"] < current[1]["total_cost"]:
                best_by_product[row["product_id"]] = (row, metrics)
        if not best_by_product:
            response["answer"] = "No feasible paths remain under the scenario constraints."
        else:
            lines = []
            for _, (row, metrics) in best_by_product.items():
                lines.append(
                    f"{row['product_name']}: best path via {row['supplier_name']} -> {row['factory_name']} "
                    f"-> {row['port_name']} totals ${metrics['total_cost']:.2f} and {metrics['total_time']} days."
                )
            response["answer"] = "\n".join(lines)

    response["trace"] = [
        {
            "row_id": row["row_id"],
            "product": row["product_name"],
            "component": row["component_name"],
            "supplier": row["supplier_name"],
            "factory": row["factory_name"],
            "port": row["port_name"],
            "market": row["market_country"],
            "reason": reason,
        }
        for row, reason in exclusions
    ]
    response["trace_summary"] = (
        f"Excluded {len(exclusions)} paths. Remaining paths: {len(filtered_rows)}."
    )
    return response

