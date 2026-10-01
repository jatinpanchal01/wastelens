"""Local SQLite persistence for the single-workspace WasteLens MVP."""
from __future__ import annotations

import json
import io
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _db_path() -> Path:
    configured = os.environ.get("WASTELENS_DB_PATH")
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parent / ".wastelens" / "wastelens.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS workspace (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            dataset_csv TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            baseline_json TEXT,
            scenario_json TEXT,
            overrides_json TEXT NOT NULL,
            current_plan_json TEXT NOT NULL,
            override_reason TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS forecast_runs (
            run_id TEXT PRIMARY KEY,
            request_key TEXT,
            dataset_id TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            target_date TEXT NOT NULL,
            meal_type TEXT NOT NULL,
            demand REAL NOT NULL,
            waste_kg REAL NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS forecast_runs_generated ON forecast_runs(generated_at DESC);
        """
    )
    run_columns = {row["name"] for row in connection.execute("PRAGMA table_info(forecast_runs)")}
    if "request_key" not in run_columns:
        connection.execute("ALTER TABLE forecast_runs ADD COLUMN request_key TEXT")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS forecast_runs_request_key ON forecast_runs(request_key) WHERE request_key IS NOT NULL")
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _decode_run(value: str | None) -> dict | None:
    if not value:
        return None
    run = json.loads(value)
    if run.get("inputs", {}).get("target_date"):
        run["inputs"]["target_date"] = date.fromisoformat(run["inputs"]["target_date"])
    return run


def save_workspace(data: pd.DataFrame, metadata: dict, baseline: dict | None,
                   scenario: dict | None, overrides: dict, current_plan: dict,
                   override_reason: str) -> None:
    """Atomically save the active dataset and resumable planning state."""
    with _connection() as connection:
        connection.execute(
            """INSERT INTO workspace VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET dataset_csv=excluded.dataset_csv,
               metadata_json=excluded.metadata_json, baseline_json=excluded.baseline_json,
               scenario_json=excluded.scenario_json, overrides_json=excluded.overrides_json,
               current_plan_json=excluded.current_plan_json, override_reason=excluded.override_reason,
               updated_at=excluded.updated_at""",
            (data.to_csv(index=False), json.dumps(metadata, default=_json_default),
             json.dumps(baseline, default=_json_default) if baseline else None,
             json.dumps(scenario, default=_json_default) if scenario else None,
             json.dumps(overrides, default=_json_default), json.dumps(current_plan, default=_json_default),
             override_reason, datetime.now().astimezone().isoformat()),
        )


def load_workspace() -> dict | None:
    """Load the saved workspace; return None when this is a new installation."""
    with _connection() as connection:
        row = connection.execute("SELECT * FROM workspace WHERE id=1").fetchone()
    if row is None:
        return None
    frame = pd.read_csv(io.StringIO(row["dataset_csv"]))
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return {
        "data": frame,
        "dataset": json.loads(row["metadata_json"]),
        "baseline": _decode_run(row["baseline_json"]),
        "scenario": _decode_run(row["scenario_json"]),
        "saved_overrides": json.loads(row["overrides_json"]),
        "current_plan": json.loads(row["current_plan_json"]),
        "override_reason": row["override_reason"],
    }


def get_active_dataset_id() -> str | None:
    """Read only the current dataset identifier to detect cross-session replacement."""
    with _connection() as connection:
        row = connection.execute("SELECT metadata_json FROM workspace WHERE id=1").fetchone()
    return json.loads(row["metadata_json"])["id"] if row else None


def save_run(run: dict, payload: dict) -> dict:
    """Atomically insert/update a decision record and return the canonical run.

    The request key is unique across sessions. Taking a write reservation before
    checking it closes the race between simultaneous identical submissions.
    """
    inputs, result = run["inputs"], run["forecast"]
    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        request_key = run.get("request_key")
        if request_key:
            existing = connection.execute(
                "SELECT payload_json FROM forecast_runs WHERE request_key=?", (request_key,)
            ).fetchone()
            if existing:
                stored = json.loads(existing["payload_json"])
                canonical = stored.get("run") or run
                # Preserve the first successful result and any operator decisions
                # already recorded against that identical request.
                return _decode_run(json.dumps(canonical, default=_json_default)) or run
        connection.execute(
            """INSERT INTO forecast_runs (run_id, request_key, dataset_id, generated_at, target_date,
               meal_type, demand, waste_kg, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET request_key=excluded.request_key, payload_json=excluded.payload_json""",
            (run["id"], run.get("request_key"), run["dataset_id"], run["generated_at"], inputs["target_date"].isoformat(),
             inputs["meal_type"], float(result["demand"]),
             float(sum(item["waste_kg"] for item in result["items"])),
            json.dumps(payload, default=_json_default)),
        )
    return run


def get_run_by_request_key(key: str) -> dict | None:
    """Find the original result for a previously committed identical request."""
    with _connection() as connection:
        row = connection.execute("SELECT * FROM forecast_runs WHERE request_key=?", (key,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["payload"] = json.loads(result.pop("payload_json"))
    if result["payload"].get("run"):
        result["payload"]["run"] = _decode_run(json.dumps(result["payload"]["run"], default=_json_default))
    return result


def list_runs(limit: int = 500) -> list[dict]:
    """Return recent saved plans, newest first."""
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM forecast_runs ORDER BY generated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows]


def clear_workspace_and_runs() -> None:
    """Delete the active local dataset, resumable plan, and saved run history."""
    with _connection() as connection:
        connection.execute("PRAGMA secure_delete=ON")
        connection.execute("DELETE FROM workspace")
        connection.execute("DELETE FROM forecast_runs")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("VACUUM")
