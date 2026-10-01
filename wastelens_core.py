"""Shared data validation and forecasting logic for WasteLens."""
from __future__ import annotations

import hashlib
import io
import json
import platform
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from wastelens_config import (MAX_UPLOAD_BYTES, MIN_HISTORY_DAYS, DEGRADED_HISTORY_DAYS,
                              DEFAULT_MODERATE_RISK_RATE, DEFAULT_HIGH_RISK_RATE,
                              MAX_EXPECTED_CUSTOMERS)

REQUIRED = ["date", "meal_type", "menu_item", "meals_prepared", "meals_served", "food_waste_kg"]
OPTIONAL = ["temperature", "holiday", "event", "expected_customers", "portion_mass_kg"]
NUMERIC = ["meals_prepared", "meals_served", "food_waste_kg", *OPTIONAL]
MAX_BYTES = MAX_UPLOAD_BYTES


def _new_regressor(algorithm: str):
    if algorithm == "xgboost":
        return XGBRegressor(n_estimators=180, max_depth=4, learning_rate=.05,
                            subsample=.9, colsample_bytree=.9, reg_lambda=1.0,
                            objective="reg:squarederror", random_state=42, n_jobs=1,
                            verbosity=0)
    return RandomForestRegressor(n_estimators=120, min_samples_leaf=2,
                                 random_state=42, n_jobs=1)
ALIASES = {
    "service_date": "date", "meal": "meal_type", "meal_period": "meal_type",
    "item": "menu_item", "item_name": "menu_item", "prepared": "meals_prepared",
    "portions_prepared": "meals_prepared", "served": "meals_served",
    "portions_served": "meals_served", "waste_kg": "food_waste_kg",
    "food_waste": "food_waste_kg", "customers": "expected_customers",
    "portion_weight_kg": "portion_mass_kg", "food_waste_unit": "waste_unit",
}


@dataclass
class ValidationResult:
    frame: pd.DataFrame | None
    errors: list[str]
    warnings: list[str]
    rejected: pd.DataFrame
    valid_rows: int
    total_rows: int
    checksum: str | None
    column_mappings: dict[str, str] | None = None
    warning_rows: int = 0


def validate_csv(raw: bytes, *, keep_last_duplicates: bool = False) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    rejected = pd.DataFrame()
    if len(raw) > MAX_BYTES:
        return ValidationResult(None, ["Upload one CSV file up to 10 MB."], [], rejected, 0, 0, None, {})
    checksum = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
        frame = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
        return ValidationResult(None, ["Could not read this file as a UTF-8 CSV. Check the encoding and delimiter, then retry."], [], rejected, 0, 0, checksum, {})
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    mappings: dict[str, str] = {}
    for alias, canonical in ALIASES.items():
        if alias in frame.columns:
            if canonical in frame.columns:
                return ValidationResult(None, [f"Both '{alias}' and '{canonical}' map to the same field. Keep one column to avoid ambiguous data."], [], rejected, 0, len(frame), checksum, {})
            frame = frame.rename(columns={alias: canonical})
            mappings[alias] = canonical
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        return ValidationResult(None, [f"Missing required column(s): {', '.join(missing)}"], [], rejected, 0, len(frame), checksum, mappings)
    if mappings:
        warnings.append("Column aliases mapped: " + ", ".join(f"{source} → {target}" for source, target in mappings.items()))
    optional_missing = {
        col: (len(frame) if col not in frame else int(frame[col].isna().sum()))
        for col in OPTIONAL
    }
    row_warning = pd.Series(False, index=frame.index, dtype=bool)
    for col in OPTIONAL:
        if col not in frame:
            frame[col] = np.nan
        row_warning |= frame[col].isna()
    if optional_missing["temperature"]:
        warnings.append(f"Temperature was missing in {optional_missing['temperature']} row(s); historical median imputation will be used for model features.")
    for col in ["holiday", "event"]:
        if optional_missing[col]:
            warnings.append(f"{optional_missing[col]} missing {col} value(s) default to 0 (no {col}).")
    if optional_missing["expected_customers"]:
        warnings.append(f"{optional_missing['expected_customers']} missing expected-customer value(s) are estimated from the maximum item servings for that date and meal.")
    if optional_missing["portion_mass_kg"]:
        warnings.append("Portion mass is missing for some or all rows; waste-risk percentages will be unavailable for those items.")
    if "waste_unit" not in frame:
        frame["waste_unit"] = "kg"
    frame["waste_unit"] = frame["waste_unit"].astype("string").str.strip().str.lower()
    original_dates = frame["date"].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for col in ["meal_type", "menu_item"]:
        frame[col] = frame[col].astype("string").str.strip()
        frame.loc[frame[col].isin(["", "<NA>", "nan", "None"]), col] = pd.NA
    numeric_parse_errors: dict[str, pd.Series] = {}
    for col in NUMERIC:
        raw_values = frame[col].astype("string").str.strip()
        comma_decimal = raw_values.str.contains(",", regex=False, na=False) & ~raw_values.str.contains(".", regex=False, na=False)
        normalized = raw_values.where(~comma_decimal, raw_values.str.replace(",", ".", regex=False))
        parsed = pd.to_numeric(normalized, errors="coerce")
        numeric_parse_errors[col] = raw_values.notna() & parsed.isna()
        frame[col] = parsed
    issue_lists: dict[int, list[str]] = {int(index): [] for index in frame.index}
    for index in frame.index:
        if pd.isna(frame.at[index, "date"]): issue_lists[int(index)].append(f"Invalid or missing date: {original_dates.at[index]}")
        if pd.notna(frame.at[index, "date"]) and frame.at[index, "date"] > pd.Timestamp(date.today()): issue_lists[int(index)].append("Future service date is not allowed")
        for col in ["meal_type", "menu_item"]:
            if pd.isna(frame.at[index, col]): issue_lists[int(index)].append(f"Missing {col}")
        for col in REQUIRED:
            if col in NUMERIC and (pd.isna(frame.at[index, col]) or numeric_parse_errors[col].at[index]):
                issue_lists[int(index)].append(f"Missing or non-numeric {col}")
        for col in NUMERIC:
            value = frame.at[index, col]
            if pd.notna(value) and not np.isfinite(value): issue_lists[int(index)].append(f"Non-finite {col}")
        for col in ["meals_prepared", "meals_served", "food_waste_kg"]:
            if pd.notna(frame.at[index, col]) and frame.at[index, col] < 0: issue_lists[int(index)].append(f"Negative {col}")
        if pd.notna(frame.at[index, "expected_customers"]) and frame.at[index, "expected_customers"] < 0: issue_lists[int(index)].append("Negative expected_customers")
        for count_column in ["meals_prepared", "meals_served", "expected_customers"]:
            value = frame.at[index, count_column]
            if pd.notna(value) and not np.isclose(value, np.round(value)): issue_lists[int(index)].append(f"{count_column} must be a whole number")
        if pd.notna(frame.at[index, "portion_mass_kg"]) and frame.at[index, "portion_mass_kg"] <= 0: issue_lists[int(index)].append("portion_mass_kg must be positive")
        if pd.notna(frame.at[index, "meals_prepared"]) and pd.notna(frame.at[index, "meals_served"]) and frame.at[index, "meals_prepared"] < frame.at[index, "meals_served"]:
            issue_lists[int(index)].append("meals_prepared is less than meals_served")
        for col in ["holiday", "event"]:
            if pd.notna(frame.at[index, col]) and frame.at[index, col] not in (0, 1): issue_lists[int(index)].append(f"{col} must be 0 or 1")
        unit = frame.at[index, "waste_unit"]
        if pd.isna(unit) or unit not in {"kg", "kilogram", "kilograms", "g", "gram", "grams"}:
            issue_lists[int(index)].append(f"Unsupported waste unit '{unit}'; convert to kg or g and identify it in waste_unit")
    grams = frame["waste_unit"].isin(["g", "gram", "grams"])
    if grams.any():
        row_warning |= grams
        warnings.append(f"{int(grams.sum())} row(s) reported waste in grams were converted to kilograms for analysis.")
    frame.loc[grams, "food_waste_kg"] = frame.loc[grams, "food_waste_kg"] / 1000
    frame["waste_unit"] = "kg"
    invalid = pd.Series({index: bool(issues) for index, issues in issue_lists.items()}, dtype=bool).reindex(frame.index, fill_value=False)
    keys = ["date", "meal_type", "menu_item"]
    duplicate = frame.duplicated(keys, keep=False) & frame[keys].notna().all(axis=1) & ~invalid
    keep_mask = pd.Series(True, index=frame.index)
    if duplicate.any():
        if keep_last_duplicates:
            duplicate_valid = frame.loc[~invalid].duplicated(keys, keep="last")
            keep_mask.loc[duplicate_valid.index] = ~duplicate_valid
            duplicate_groups = frame.loc[duplicate].groupby(keys, dropna=False).size()
            discarded_count = int((duplicate_groups - 1).sum())
            warnings.append(f"{discarded_count} earlier duplicate row(s) were discarded after explicit confirmation; the last valid row per key was kept.")
        else:
            errors.append(f"Found {int(duplicate.sum())} rows with duplicate date/meal/item keys. Keep the last row only after confirmation, or resolve duplicates in the CSV.")
    rejected_mask = invalid | (duplicate & (~keep_mask if keep_last_duplicates else True))
    rejected = frame.loc[rejected_mask].copy()
    issues = []
    for index in rejected.index:
        row_issues = issue_lists[int(index)].copy()
        if duplicate.at[index]:
            row_issues.append("Duplicate date/meal/item key; earlier row discarded" if keep_last_duplicates else "Duplicate date/meal/item key; resolve or confirm keep-last")
        issues.append("; ".join(row_issues))
    rejected["_validation_issue"] = issues
    clean = frame.loc[~invalid & keep_mask].copy()
    warning_rows = int((row_warning & ~invalid & keep_mask).sum())
    total = len(frame)
    if total == 0:
        errors.append("The CSV has no data rows.")
    if total and len(clean) / total < 0.8:
        errors.append(f"Only {len(clean)} of {total} rows are valid; at least 80% must remain.")
    days = clean["date"].nunique()
    if days < MIN_HISTORY_DAYS:
        errors.append(f"At least 14 usable history days are required; found {days}.")
    elif days < DEGRADED_HISTORY_DAYS:
        warnings.append(f"Degraded forecast: only {days} usable history days (14–29).")
    if len(rejected):
        warnings.append(f"{len(rejected)} row(s) were rejected. Download the row-level report to review each issue.")
    if errors:
        clean_result = clean if len(clean) else None
    else:
        clean_result = clean
    return ValidationResult(clean_result, errors, warnings, rejected, len(clean), total, checksum, mappings, warning_rows)


def request_key(dataset_id: str, model_version: str, inputs: dict, buffer: float, increment: int,
                item_settings: dict | None = None, risk_thresholds: dict | None = None) -> str:
    """Stable idempotency fingerprint for the same dataset/model/forecast request."""
    payload = {
        "dataset_id": dataset_id,
        "model_version": model_version,
        "target_date": pd.Timestamp(inputs["target_date"]).date().isoformat(),
        "meal_type": str(inputs["meal_type"]),
        "expected_customers": int(inputs["expected_customers"]),
        "temperature": float(inputs["temperature"]),
        "is_holiday": bool(inputs["is_holiday"]),
        "has_event": bool(inputs["has_event"]),
        "menu_items": sorted(map(str, inputs["menu_items"])),
        "buffer": float(buffer),
        "increment": int(increment),
        "item_settings": item_settings or {},
        "risk_thresholds": risk_thresholds or {},
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"]).dt.normalize()
    defaults = {"temperature": x["temperature"].median() if "temperature" in x and x["temperature"].notna().any() else 25.0,
                "holiday": 0, "event": 0, "expected_customers": np.nan}
    for col, default in defaults.items():
        if col not in x: x[col] = default
        if col != "expected_customers": x[col] = pd.to_numeric(x[col], errors="coerce").fillna(default)
    # Customer counts are logged per date/meal, so medians avoid multiplying by item count.
    x["expected_customers"] = x["expected_customers"].fillna(x.groupby(["date", "meal_type"])["meals_served"].transform("max"))
    x["day_of_week"] = x["date"].dt.dayofweek
    return x


def _demand_rows(df: pd.DataFrame) -> pd.DataFrame:
    x = _normalize(df)
    keys = ["date", "meal_type"]
    agg = x.groupby(keys, as_index=False).agg(
        meals_served=("meals_served", "max"), expected_customers=("expected_customers", "median"),
        temperature=("temperature", "median"), holiday=("holiday", "max"), event=("event", "max"),
        day_of_week=("day_of_week", "first"))
    return agg


def _add_temporal_features(frame: pd.DataFrame, target: str, groups: list[str]) -> pd.DataFrame:
    """Add only prior-observation lags and rolling values within each time series."""
    x = frame.copy().reset_index(drop=True).sort_values([*groups, "date"], kind="stable")
    grouped = x.groupby(groups, dropna=False, sort=False)[target]
    prefix = target
    x[f"lag_{prefix}_1"] = grouped.shift(1)
    x[f"lag_{prefix}_7"] = grouped.shift(7)
    x[f"rolling_{prefix}_7"] = grouped.transform(lambda values: values.shift(1).rolling(7, min_periods=1).mean())
    return x


def _inference_features(inputs: dict, history: pd.DataFrame, target: str,
                        item: str | None = None) -> pd.DataFrame:
    """Create inference features from observations strictly before the forecast date."""
    normalized = _normalize(history)
    target_date = pd.Timestamp(inputs["target_date"]).normalize()
    if target == "meals_served":
        observed = _demand_rows(normalized)
        observed = observed[(observed.meal_type.astype(str) == inputs["meal_type"]) & (observed.date < target_date)]
        query = pd.DataFrame([{
            # The current target value is never used by its lag/rolling columns;
            # a numeric placeholder avoids dtype ambiguity in pandas concat.
            "date": target_date, "meal_type": inputs["meal_type"], "meals_served": 0.0,
            "expected_customers": inputs["expected_customers"], "temperature": inputs["temperature"],
            "holiday": int(inputs["is_holiday"]), "event": int(inputs["has_event"]),
            "day_of_week": target_date.dayofweek,
        }])
        combined = _add_temporal_features(pd.concat([observed, query], ignore_index=True), target, ["meal_type"])
    else:
        observed = normalized[(normalized.menu_item.astype(str) == str(item)) &
                              (normalized.meal_type.astype(str) == inputs["meal_type"]) &
                              (normalized.date < target_date)]
        query = pd.DataFrame([{
            "date": target_date, "meal_type": inputs["meal_type"], "menu_item": item,
            "food_waste_kg": 0.0, "expected_customers": inputs["expected_customers"],
            "temperature": inputs["temperature"], "holiday": int(inputs["is_holiday"]),
            "event": int(inputs["has_event"]), "day_of_week": target_date.dayofweek,
        }])
        combined = _add_temporal_features(pd.concat([observed, query], ignore_index=True), target,
                                          ["menu_item", "meal_type"])
    return _design(combined.tail(1))


def _design(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    x = df.copy()
    x["is_weekend"] = (x["day_of_week"] >= 5).astype(int)
    x = pd.get_dummies(x, columns=[c for c in ["meal_type", "menu_item"] if c in x], dtype=float)
    # Keep training and inference on the same information available before service.
    # In particular, never pass prepared/served totals or waste outcomes as features.
    valid = ["expected_customers", "temperature", "holiday", "event", "day_of_week", "is_weekend"]
    valid += [c for c in x.columns if c.startswith(("meal_type_", "menu_item_", "lag_", "rolling_"))]
    x = x.reindex(columns=valid, fill_value=0)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0)
    return x


def _fit_eval(train: pd.DataFrame, test: pd.DataFrame, target: str, item_level: bool = False,
              algorithm_override: str | None = None):
    train = _normalize(train)
    test = _normalize(test)
    train_dates = set(train.date)
    test_dates = set(test.date)
    combined = pd.concat([train, test], ignore_index=True).sort_values("date")
    if not item_level:
        combined = _demand_rows(combined)
    group_columns = ["menu_item", "meal_type"] if item_level else ["meal_type"]
    combined = _add_temporal_features(combined, target, group_columns)
    train = combined[combined.date.isin(train_dates)].copy()
    test = combined[combined.date.isin(test_dates)].copy()
    Xtr = _design(train).drop(columns=["date", target], errors="ignore")
    Xte = _design(test, list(Xtr.columns)).drop(columns=["date", target], errors="ignore")
    ytr, yte = train[target].astype(float), test[target].astype(float)
    candidate_results = {}
    for algorithm in ["random_forest", "xgboost"]:
        model_candidate = _new_regressor(algorithm)
        fit_start = time.perf_counter()
        model_candidate.fit(Xtr, ytr)
        fit_ms = (time.perf_counter() - fit_start) * 1000
        predict_start = time.perf_counter()
        candidate_pred = np.maximum(0, model_candidate.predict(Xte))
        predict_ms = (time.perf_counter() - predict_start) * 1000
        fold_mae = [mean_absolute_error(yte.iloc[indexes], candidate_pred[indexes])
                    for indexes in np.array_split(np.arange(len(yte)), min(3, len(yte))) if len(indexes)]
        candidate_results[algorithm] = {
            "model": model_candidate,
            "pred": candidate_pred,
            "mae": float(mean_absolute_error(yte, candidate_pred)),
            "stability_mae_std": float(np.std(fold_mae)) if len(fold_mae) > 1 else 0.0,
            "fit_ms": fit_ms,
            "inference_ms": predict_ms,
        }
    algorithm = algorithm_override or min(candidate_results, key=lambda name: (
        candidate_results[name]["mae"], candidate_results[name]["stability_mae_std"],
        candidate_results[name]["inference_ms"]))
    model = candidate_results[algorithm]["model"]
    pred = candidate_results[algorithm]["pred"]
    baseline_groups = ["menu_item", "meal_type", "day_of_week"] if item_level else ["meal_type", "day_of_week"]
    baseline_groups = [column for column in baseline_groups if column in train and column in test]
    grouped_medians = train.groupby(baseline_groups, dropna=False)[target].median() if baseline_groups else pd.Series(dtype=float)
    fallback = float(ytr.median())
    baseline_values = []
    for _, row in test.iterrows():
        key = tuple(row[column] for column in baseline_groups)
        if len(key) == 1: key = key[0]
        baseline_values.append(float(grouped_medians.get(key, fallback)))
    baseline = np.asarray(baseline_values, dtype=float)
    metrics = {"mae": float(mean_absolute_error(yte, pred)), "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
               "r2": float(r2_score(yte, pred)) if len(yte) > 1 else 0.0,
               "wape": float(np.abs(yte.to_numpy() - pred).sum() / np.abs(yte.to_numpy()).sum()) if np.abs(yte.to_numpy()).sum() else None,
               "baseline_mae": float(mean_absolute_error(yte, baseline)), "baseline_method": "grouped historical median" if baseline_groups else "training median fallback", "rows": len(yte),
               "algorithm": algorithm, "selection_basis": "lowest chronological holdout MAE; ties by three-block MAE stability then inference time",
               "model_comparison": {name: {key: value for key, value in entry.items() if key not in {"model", "pred"}}
                                    for name, entry in candidate_results.items()}}
    metrics["slices"] = {}
    slice_columns = [column for column in ["meal_type", "day_of_week", "menu_item"] if column in test]
    for column in slice_columns:
        candidates = test[column].dropna().unique()
        if column == "menu_item" and "meals_served" in train:
            top_items = train.groupby("menu_item").meals_served.sum().nlargest(5).index
            candidates = [value for value in candidates if value in top_items]
        by_slice = []
        for value in candidates:
            mask = test[column].to_numpy() == value
            if mask.any():
                by_slice.append({"slice": str(value), "rows": int(mask.sum()), "mae": float(mean_absolute_error(yte.to_numpy()[mask], pred[mask]))})
        metrics["slices"][column] = by_slice
    return model, list(Xtr.columns), metrics, algorithm


def train_models(df: pd.DataFrame, algorithms: dict | None = None) -> dict:
    algorithms = algorithms or {}
    normalized = _normalize(df)
    unique_dates = sorted(normalized["date"].dropna().unique())
    split = max(1, int(len(unique_dates) * 0.8))
    train_dates, test_dates = set(unique_dates[:split]), set(unique_dates[split:])
    if not test_dates:
        raise ValueError("At least two distinct dates are needed to evaluate a model.")
    train = normalized[normalized.date.isin(train_dates)]
    test = normalized[normalized.date.isin(test_dates)]
    eval_demand = _fit_eval(train, test, "meals_served", algorithm_override=algorithms.get("demand"))
    eval_waste = _fit_eval(train, test, "food_waste_kg", item_level=True, algorithm_override=algorithms.get("waste"))
    # Final models train on all accepted rows; the independent date holdout above remains the reported evaluation.
    demand_data = _add_temporal_features(_demand_rows(normalized), "meals_served", ["meal_type"])
    Xd = _design(demand_data)
    demand_algorithm = eval_demand[3]
    demand_model = _new_regressor(demand_algorithm).fit(Xd, demand_data.meals_served)
    waste_data = _add_temporal_features(normalized, "food_waste_kg", ["menu_item", "meal_type"])
    Xw = _design(waste_data)
    waste_algorithm = eval_waste[3]
    waste_model = _new_regressor(waste_algorithm).fit(Xw, waste_data.food_waste_kg)
    return {"demand_model": demand_model, "demand_features": list(Xd.columns), "waste_model": waste_model,
            "waste_features": list(Xw.columns), "demand_metrics": eval_demand[2], "waste_metrics": eval_waste[2],
            "demand_algorithm": demand_algorithm, "waste_algorithm": waste_algorithm,
            "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                                 "xgboost": xgboost.__version__},
            "quality": "baseline-beating" if eval_demand[2]["mae"] < eval_demand[2]["baseline_mae"] and eval_waste[2]["mae"] < eval_waste[2]["baseline_mae"] else "weak-baseline",
            "date_min": normalized.date.min(), "date_max": normalized.date.max()}


def make_features(inputs: dict, feature_names: list[str], item: str | None = None,
                  history: pd.DataFrame | None = None, target: str = "meals_served") -> pd.DataFrame:
    if history is not None:
        temporal_target = "meals_served" if target == "meals_served" else "food_waste_kg"
        return _inference_features(inputs, history, temporal_target, item).reindex(columns=feature_names, fill_value=0)
    row = {"expected_customers": inputs["expected_customers"], "temperature": inputs["temperature"],
           "holiday": int(inputs["is_holiday"]), "event": int(inputs["has_event"]),
           "day_of_week": inputs["target_date"].weekday(), "meal_type": inputs["meal_type"]}
    if item is not None: row["menu_item"] = item
    return _design(pd.DataFrame([row]), feature_names)


def forecast(models: dict, history: pd.DataFrame, inputs: dict) -> dict:
    customers = inputs.get("expected_customers")
    try:
        valid_customers = (float(customers).is_integer() and 0 <= int(customers) <= MAX_EXPECTED_CUSTOMERS)
    except (TypeError, ValueError, OverflowError):
        valid_customers = False
    if not valid_customers:
        return {"error": f"Enter expected customers from 0 to {MAX_EXPECTED_CUSTOMERS:,}."}
    try:
        temperature_value = float(inputs.get("temperature", np.nan))
    except (TypeError, ValueError, OverflowError):
        temperature_value = np.nan
    if not np.isfinite(temperature_value):
        return {"error": "Enter a finite temperature value."}
    known_meals = set(history.meal_type.astype(str).unique())
    if inputs["meal_type"] not in known_meals:
        return {"error": f"No history for meal type {inputs['meal_type']}; choose an observed meal type."}
    historical_items = set(history.menu_item.astype(str).unique())
    requested = [i for i in inputs["menu_items"] if i in historical_items]
    excluded = [i for i in inputs["menu_items"] if i not in historical_items]
    if not requested:
        return {"error": "Select at least one menu item with historical data."}
    raw_demand = 0.0 if inputs["expected_customers"] == 0 else float(models["demand_model"].predict(make_features(inputs, models["demand_features"], history=history))[0])
    if not np.isfinite(raw_demand):
        return {"error": "The demand model returned an invalid value. Retry after checking the model and input data."}
    demand = max(0.0, raw_demand)
    items = []
    postprocessing_warnings = []
    if raw_demand < 0:
        postprocessing_warnings.append("Raw demand prediction was negative and was clipped to zero.")
    waste_feature_rows = []
    clean = _normalize(history)
    meal_history = clean[clean.meal_type.astype(str) == inputs["meal_type"]]
    reference_served = meal_history.groupby(["date", "meal_type"])["meals_served"].max()
    for item in requested:
        item_features = make_features(inputs, models["waste_features"], item, history, "food_waste_kg")
        waste_feature_rows.append(item_features)
        raw_waste = 0.0 if inputs["expected_customers"] == 0 else float(models["waste_model"].predict(item_features)[0])
        if not np.isfinite(raw_waste):
            return {"error": "The waste model returned an invalid value. Retry after checking the model and input data."}
        waste = max(0.0, raw_waste)
        if raw_waste < 0:
            postprocessing_warnings.append("A raw item-waste prediction was negative and was clipped to zero.")
        subset = meal_history[meal_history.menu_item.astype(str) == item]
        reference = subset.set_index(["date", "meal_type"]).index.map(reference_served)
        reference_values = np.asarray(reference, dtype=float)
        ratios = pd.Series(subset.meals_served.to_numpy() / np.where(reference_values == 0, np.nan, reference_values), index=subset.index)
        serving_ratio = float(ratios.replace([np.inf, -np.inf], np.nan).dropna().median()) if ratios.notna().any() else None
        mass_series = pd.to_numeric(subset.get("portion_mass_kg", pd.Series(dtype=float)), errors="coerce").dropna()
        portion_mass = float(mass_series.median()) if len(mass_series) else None
        items.append({"item": item, "waste_kg": waste, "raw_waste_kg": raw_waste,
                      "serving_ratio": serving_ratio, "portion_mass_kg": portion_mass})
    if inputs["expected_customers"] == 0:
        item_explanations = [{"kind": "not applicable (zero-customer plan)", "drivers": []} for _ in items]
    else:
        item_explanations = explain_batch(models["waste_model"], models["waste_features"], pd.concat(waste_feature_rows, ignore_index=True))
    for item, explanation in zip(items, item_explanations):
        item["explanation"] = explanation
    warnings = []
    warnings.extend(postprocessing_warnings)
    if demand > inputs["expected_customers"]:
        warnings.append(f"Predicted demand ({demand:.0f}) exceeds the entered customer count ({inputs['expected_customers']}); review attendance and treat this forecast as low confidence.")
    for col, value in [("expected_customers", inputs["expected_customers"]), ("temperature", inputs["temperature"])]:
        observed = pd.to_numeric(history[col], errors="coerce").dropna() if col in history else pd.Series(dtype=float)
        if len(observed) and (value < observed.min() or value > observed.max()):
            warnings.append(f"{col.replace('_', ' ').title()} is outside the training range ({observed.min():.1f}–{observed.max():.1f}); confidence is lower.")
    return {"demand": demand, "raw_demand": raw_demand, "items": items, "excluded_items": excluded, "warnings": warnings,
            "explanation": explain_demand(models, inputs, history)}


def recommendations(result: dict, buffer: float = .05, increment: int = 1,
                     item_settings: dict | None = None,
                     risk_thresholds: dict | None = None) -> pd.DataFrame:
    predicted_demand = max(0.0, float(result["demand"]))
    demand = int(np.ceil(predicted_demand))
    recommended_meals = int(np.ceil(predicted_demand * (1 + buffer)))
    item_settings = item_settings or {}
    default_policy = risk_thresholds is None
    risk_thresholds = risk_thresholds or {"moderate": DEFAULT_MODERATE_RISK_RATE, "high": DEFAULT_HIGH_RISK_RATE}
    moderate_threshold = float(risk_thresholds["moderate"])
    high_threshold = float(risk_thresholds["high"])
    if not 0 <= moderate_threshold < high_threshold <= 1:
        raise ValueError("Waste risk thresholds must satisfy 0 ≤ moderate < high ≤ 1.")
    rows = []
    for item in result["items"]:
        ratio = item["serving_ratio"]
        forecast_portions = int(np.ceil(demand * ratio)) if ratio is not None else None
        settings = item_settings.get(item["item"], {})
        item_increment = max(1, int(settings.get("increment", increment)))
        capacity = settings.get("capacity")
        capacity = int(capacity) if capacity is not None and pd.notna(capacity) and int(capacity) > 0 else None
        raw_recommendation = int(np.ceil((recommended_meals * ratio) / item_increment) * item_increment) if ratio is not None else None
        recommended = min(raw_recommendation, capacity) if raw_recommendation is not None and capacity is not None else raw_recommendation
        mass = item["portion_mass_kg"]
        waste_rate = item["waste_kg"] / (recommended * mass) if recommended and mass else None
        risk = "Not available" if waste_rate is None else ("High" if waste_rate > high_threshold else "Moderate" if waste_rate >= moderate_threshold else "Low")
        unmet = "Not assessed" if capacity is None else "High" if forecast_portions is not None and capacity < forecast_portions else "Within capacity"
        explanation = item.get("explanation", {})
        drivers = "; ".join(
            f"{driver['feature'].replace('_', ' ').title()} {driver['impact']:+.2f} kg" if explanation.get("kind") == "local SHAP"
            else f"{driver['feature'].replace('_', ' ').title()} ({driver['impact']:.1%} global importance)"
            for driver in explanation.get("drivers", [])
        )
        rows.append({"Menu Item": item["item"], "Forecast portions": forecast_portions if forecast_portions is not None else "Not available",
                     "Recommended portions": recommended if recommended is not None else "Not available",
                     "Uncapped recommendation": raw_recommendation if raw_recommendation is not None else "Not available",
                     "Capacity portions": capacity if capacity is not None else "Not configured",
                     "Unmet demand risk": unmet,
                     "Risk policy": "Default policy" if default_policy else f"Moderate ≥ {moderate_threshold:.0%}; High > {high_threshold:.0%}",
                     "Portion mass (kg)": round(mass, 3) if mass is not None else "Not provided",
                     "Predicted waste (kg)": round(item["waste_kg"], 2), "Waste risk": risk, "Waste drivers": drivers})
    return pd.DataFrame(rows)


def explain_demand(models: dict, inputs: dict, history: pd.DataFrame | None = None) -> dict:
    """Return per-run SHAP drivers, or an explicitly labeled global fallback."""
    x = make_features(inputs, models["demand_features"], history=history)
    return explain_batch(models["demand_model"], models["demand_features"], x)[0]


def explain_batch(model, feature_names: list[str], features: pd.DataFrame) -> list[dict]:
    """Explain one or more predictions, with an explicit global-importance fallback."""
    try:
        import shap
        values = np.asarray(shap.TreeExplainer(model).shap_values(features))
        if values.ndim == 3 and values.shape[-1] == 1: values = values[:, :, 0]
        results = []
        for row in values:
            impacts = [{"feature": feature_names[i], "impact": float(row[i])} for i in range(len(row))]
            increasing = sorted((entry for entry in impacts if entry["impact"] > 0), key=lambda entry: entry["impact"], reverse=True)[:3]
            decreasing = sorted((entry for entry in impacts if entry["impact"] < 0), key=lambda entry: entry["impact"])[:3]
            results.append({"kind": "local SHAP", "drivers": increasing + decreasing})
        return results
    except Exception:
        importance = model.feature_importances_
        indexes = np.argsort(importance)[-3:][::-1]
        fallback = {"kind": "global feature-importance fallback", "drivers": [{"feature": feature_names[i], "impact": float(importance[i])} for i in indexes]}
        return [fallback.copy() for _ in range(len(features))]
