from __future__ import annotations

from pathlib import Path
from typing import Any

from duckduckgo_search.exceptions import DuckDuckGoSearchException
from openpyxl import Workbook


def create_excel(filename: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    workbook = Workbook()
    worksheet = workbook.active

    headers = sorted({key for row in rows for key in row})
    if headers:
        worksheet.append(headers)
        for row in rows:
            worksheet.append([row.get(header) for header in headers])

    output_path = Path(filename).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)

    return {
        "tool": "create_excel",
        "file_path": str(output_path),
        "rows_written": len(rows),
        "columns": headers,
    }


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except (DuckDuckGoSearchException, OSError, RuntimeError, ValueError) as exc:
        return {"tool": "web_search", "query": query, "results": [], "count": 0, "error": str(exc)}

    normalized = [
        {
            "title": item.get("title", ""),
            "url": item.get("href", ""),
            "snippet": item.get("body", ""),
        }
        for item in results
    ]

    return {
        "tool": "web_search",
        "query": query,
        "results": normalized,
        "count": len(normalized),
    }
