"""Structured, privacy-conscious event logging for WasteLens."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone


LOGGER = logging.getLogger("wastelens")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
LOGGER.propagate = False


def trace_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(event: str, *, session_id: str, dataset_id: str | None = None,
              model_version: str | None = None, duration_ms: float | None = None,
              status: str = "success", error_code: str | None = None,
              trace: str | None = None, run_id: str | None = None,
              model_outputs: dict | None = None) -> str:
    """Log stable operational fields only; never log uploaded rows or menu data."""
    identifier = trace or trace_id()
    fields = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "trace_id": identifier,
        "session_id": session_id,
        "dataset_id": dataset_id,
        "model_version": model_version,
        "duration_ms": round(float(duration_ms), 2) if duration_ms is not None else None,
        "status": status,
        "error_code": error_code,
        "run_id": run_id,
        "model_outputs": model_outputs,
    }
    LOGGER.info(json.dumps(fields, separators=(",", ":"), ensure_ascii=False))
    return identifier
