# Worlds Apart

Worlds Apart shows what happens when two AIs see the same data — but only one has a world model.

## Overview

This demo compares Neural Recall (RAG) with a Symbolic World Model (GraphRAG). Both systems ingest the same uploaded spreadsheet, rebuild all derived artifacts on upload, and answer the same questions under shared scenario controls.

## Requirements

- Python 3.11+
- Install dependencies: `pip install -r requirements.txt`

## Run the app

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000` and upload `supply_chain.csv` or an `.xlsx` equivalent.

## API

- `POST /data/upload` — upload CSV/XLSX, rebuilds graph, notes, and vector index.
- `POST /chat/rag` — RAG response using retrieved notes.
- `POST /chat/graphrag` — GraphRAG response with reasoning trace.
- `POST /chat/both` — returns both responses.
