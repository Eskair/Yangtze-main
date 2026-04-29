"""Shared run context helpers for canonical pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT / "data"
CONTEXT_DIR = DATA_DIR / "context"
CONTEXT_PATH = CONTEXT_DIR / "context.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_context() -> Dict[str, Any]:
    if not CONTEXT_PATH.exists():
        return {}
    try:
        return json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_context(ctx: Dict[str, Any]) -> None:
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_PATH.write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")


def init_context(proposal_id: str, source_path: Optional[str] = None) -> Dict[str, Any]:
    ctx = {
        "schema": "yangtze.run_context.v1",
        "canonical_pipeline": "run_pipeline.py",
        "proposal_id": proposal_id,
        "source_path": source_path or "",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "completed_stages": [],
    }
    save_context(ctx)
    return ctx


def get_context_proposal_id() -> str:
    ctx = load_context()
    return str(ctx.get("proposal_id") or "").strip()


def mark_stage(stage_name: str, extra: Optional[Dict[str, Any]] = None) -> None:
    ctx = load_context()
    completed = ctx.get("completed_stages")
    if not isinstance(completed, list):
        completed = []
    if stage_name not in completed:
        completed.append(stage_name)
    ctx["completed_stages"] = completed
    ctx["updated_at"] = _utc_now()
    if isinstance(extra, dict) and extra:
        stage_meta = ctx.get("stage_meta")
        if not isinstance(stage_meta, dict):
            stage_meta = {}
        stage_meta[stage_name] = extra
        ctx["stage_meta"] = stage_meta
    save_context(ctx)
