import json
from typing import Any, Dict, List, Optional


def load_knowledge_base(kb_path: str) -> List[Dict[str, Any]]:
    with open(kb_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list):
        raise ValueError("knowledge_base.json must be a JSON array")
    return docs


def retrieve_kb_context(
    docs: List[Dict[str, Any]],
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    max_docs: int = 4,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: set = set()

    for doc in docs:
        if doc.get("name", "").lower() == (genre or "").lower() and doc["id"] not in seen:
            selected.append(doc)
            seen.add(doc["id"])

    for doc in docs:
        if doc.get("name", "").lower() == (mood or "").lower() and doc["id"] not in seen:
            selected.append(doc)
            seen.add(doc["id"])

    query_tokens = set()
    if genre:
        query_tokens.update(genre.lower().split())
    if mood:
        query_tokens.update(mood.lower().split())

    if len(selected) < max_docs and query_tokens:
        for doc in docs:
            if doc["id"] in seen:
                continue
            if any(tok in doc.get("description", "").lower() for tok in query_tokens):
                selected.append(doc)
                seen.add(doc["id"])
            if len(selected) >= max_docs:
                break

    return selected[:max_docs]


def format_kb_context(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return ""
    lines = ["Knowledge Base Context:"]
    for doc in docs:
        lines.append(f"- [{doc.get('type','?')}:{doc.get('name','?')}] {doc.get('description','')}")
    return "\n".join(lines)
