"""Validation helpers for daily aggregate food-waste datasets (insights only)."""
from __future__ import annotations

import hashlib
import io
import re

import numpy as np
import pandas as pd


def read_aggregate_file(raw: bytes, filename: str) -> pd.DataFrame:
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("Upload one CSV or XLSX file up to 10 MB.")
    suffix = filename.lower().rsplit(".", 1)[-1]
    try:
        if suffix == "csv":
            frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
        elif suffix in {"xlsx", "xlsm"}:
            frame = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        else:
            raise ValueError("Choose a CSV or XLSX file.")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise ValueError("Could not read the file. Check that it is a valid UTF-8 CSV or XLSX workbook.") from error
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("The file needs a header row and at least one data column.")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def suggest_aggregate_columns(columns) -> dict[str, str | None]:
    def norm(value: str) -> str:
        value = value.strip().lower().replace("ä", "a").replace("ö", "o").replace("å", "a")
        return re.sub(r"[^a-z0-9]+", "", value)
    aliases = {
        "date": {"date", "day", "service_date", "paivamaara", "paivä", "pvm"},
        "diners": {"diners", "customers", "covers", "ruokailijat"},
        "plate_waste": {"pw", "platewaste", "platewastekg", "totalplatewastekg"},
        "kitchen_waste": {"ksw", "kitchenwaste", "servingwaste", "kitchenservingwastekg"},
        "menu": {"menu", "menuid", "lunchmenu"},
    }
    normalized = {column: norm(column) for column in columns}
    result = {}
    for key, names in aliases.items():
        result[key] = next((column for column, value in normalized.items() if value in names), None)
    return result


def normalize_aggregate(frame: pd.DataFrame, mapping: dict[str, str | None]) -> tuple[pd.DataFrame, list[str]]:
    if not mapping.get("date"):
        raise ValueError("Map a date column to continue.")
    if not mapping.get("plate_waste") and not mapping.get("kitchen_waste"):
        raise ValueError("Map at least one measured waste column (in kilograms).")
    result = pd.DataFrame(index=frame.index)
    result["date"] = pd.to_datetime(frame[mapping["date"]], errors="coerce", dayfirst=True).dt.normalize()
    warnings: list[str] = []
    for target in ("plate_waste", "kitchen_waste"):
        source = mapping.get(target)
        target_name = "plate_waste_kg" if target == "plate_waste" else "kitchen_serving_waste_kg"
        if source:
            values = pd.to_numeric(frame[source].astype("string").str.replace(",", ".", regex=False), errors="coerce")
            bad = values.isna() | ~np.isfinite(values) | (values < 0)
            if bad.any():
                raise ValueError(f"Column '{source}' has {int(bad.sum())} missing, invalid, or negative measurement(s). Correct them before import.")
            result[target_name] = values.astype(float)
        else:
            result[target_name] = np.nan
    source = mapping.get("diners")
    if source:
        diners = pd.to_numeric(frame[source].astype("string").str.replace(",", ".", regex=False), errors="coerce")
        bad = diners.isna() | ~np.isfinite(diners) | (diners <= 0)
        if bad.any():
            warnings.append(f"{int(bad.sum())} row(s) have no valid diner count; per-diner rates are unavailable for those rows.")
            diners = diners.mask(bad)
        result["diners"] = diners
    else:
        result["diners"] = np.nan
        warnings.append("No diner-count column was mapped; per-diner rates are unavailable.")
    if mapping.get("menu"):
        result["menu_id"] = frame[mapping["menu"]].astype("string")
    invalid_date = result["date"].isna()
    if invalid_date.any():
        raise ValueError(f"Date column has {int(invalid_date.sum())} missing or invalid value(s).")
    if (result["date"] > pd.Timestamp.now().normalize()).any():
        raise ValueError("Future dates are not accepted for historical insights.")
    if result["date"].duplicated().any():
        warnings.append("Multiple rows share a date. Daily totals are summed; diner counts are summed only when each row represents a separate service.")
    result["total_measured_waste_kg"] = result[["plate_waste_kg", "kitchen_serving_waste_kg"]].sum(axis=1, min_count=1)
    result["waste_kg_per_diner"] = result["total_measured_waste_kg"] / result["diners"]
    return result.reset_index(drop=True), warnings


def aggregate_checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
