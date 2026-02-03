import json
import os
import textwrap
from typing import Any, Dict, Iterable

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency
    BaseModel = None  # type: ignore[assignment]


def _make_serializable(value: Any, max_width: int = 120, key: str | None = None) -> Any:
    preserve_string_keys = {"cypher", "prompt"}
    if BaseModel is not None and isinstance(value, BaseModel):
        return _make_serializable(value.model_dump(), max_width=max_width)
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        return _make_serializable(value.model_dump(), max_width=max_width)
    if isinstance(value, dict):
        return {str(k): _make_serializable(v, max_width=max_width, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_serializable(v, max_width=max_width, key=key) for v in value]
    if isinstance(value, str):
        if key in preserve_string_keys:
            return value
        if "\n" in value:
            return value.splitlines()
        if len(value) > max_width:
            return textwrap.wrap(value, width=max_width)
        return value
    return value


def log_json(file_path: str, data_to_save: Dict) -> None:
    try:
        with open(file_path, 'w') as json_file:
            serializable = _make_serializable(data_to_save)
            json.dump(serializable, json_file, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to {file_path}")
    except IOError as e:
        print(f"Error saving file: {e}")


def log_text(file_path: str, content: str) -> None:
    try:
        with open(file_path, "w") as handle:
            handle.write(content)
        print(f"Text saved to {file_path}")
    except IOError as e:
        print(f"Error saving file: {e}")


def log_cypher_queries(file_path: str, queries: Iterable[Dict[str, Any]]) -> None:
    try:
        lines: list[str] = []
        for idx, query in enumerate(queries, start=1):
            cypher = (query.get("cypher") or "").strip()
            if not cypher:
                continue
            reason = (query.get("reason") or "").strip()
            params = query.get("params") or {}
            substituted = cypher
            if params:
                for key, value in params.items():
                    if isinstance(value, bool):
                        rendered = "true" if value else "false"
                    elif value is None:
                        rendered = "null"
                    elif isinstance(value, (int, float)):
                        rendered = str(value)
                    else:
                        escaped = str(value).replace("\\", "\\\\").replace("\"", "\\\"")
                        rendered = f'"{escaped}"'
                    substituted = substituted.replace(f"${key}", rendered)
            if reason:
                lines.append(f"// Query {idx}: {reason}")
            else:
                lines.append(f"// Query {idx}")
            if params:
                lines.append(f"// params: {json.dumps(params, ensure_ascii=False)}")
            lines.append(cypher)
            if params:
                lines.append("// substituted")
                lines.append(substituted)
            if not cypher.endswith(";"):
                lines.append(";")
            lines.append("")

        with open(file_path, "w") as cypher_file:
            cypher_file.write("\n".join(lines).rstrip() + "\n")
        print(f"Cypher queries saved to {file_path}")
    except IOError as e:
        print(f"Error saving cypher file: {e}")
