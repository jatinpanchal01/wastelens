from __future__ import annotations

import datetime as dt
import html
import io
import json
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from wastelens_core import REQUIRED, validate_csv, train_models, forecast, recommendations, request_key
from wastelens_store import clear_workspace_and_runs, get_active_dataset_id, get_run_by_request_key, list_runs, load_workspace, save_run, save_workspace
from wastelens_observability import log_event, trace_id
from wastelens_theme import PAGE_ICON, FOREST, OCHRE, inject_theme, render_empty_state, render_header, style_chart
from wastelens_config import (DEFAULT_BUFFER, DEFAULT_HIGH_RISK_RATE, DEFAULT_MODERATE_RISK_RATE,
                              DEFAULT_PRODUCTION_INCREMENT, MAX_EXPECTED_CUSTOMERS, INSIGHTS_DEFAULT_DAYS)
from wastelens_aggregate import aggregate_checksum, normalize_aggregate, read_aggregate_file, suggest_aggregate_columns

st.set_page_config(page_title="WasteLens", page_icon=PAGE_ICON, layout="wide")
def require_authenticated_user():
    """Require an allow-listed OIDC identity outside explicit local development."""
    import os

    if os.environ.get("WASTELENS_ENV", "development").lower() == "development":
        return
    try:
        auth = st.secrets["auth"]
        allowed = {email.strip().lower() for email in st.secrets["wastelens"]["allowed_emails"] if email.strip()}
        required = ("client_id", "client_secret", "redirect_uri", "cookie_secret", "server_metadata_url")
        if not allowed or any(not auth.get(key) for key in required):
            raise KeyError("Incomplete authentication configuration")
    except Exception:
        st.error("Sign-in is not configured. Set the Google OIDC values and allowed email list in Streamlit secrets before enabling production mode.")
        st.stop()
    if not st.user.is_logged_in:
        st.title("WasteLens")
        st.write("Sign in with your authorized Google account to continue.")
        if st.button("Continue with Google", type="primary"):
            st.login()
        st.stop()
    email = str(getattr(st.user, "email", "") or "").strip().lower()
    if not email or email not in allowed or getattr(st.user, "email_verified", False) is not True:
        st.error("This Google account is not authorized for this WasteLens workspace.")
        if st.button("Sign out"):
            st.logout()
        st.stop()

require_authenticated_user()
inject_theme()
render_header()
import os
if os.environ.get("WASTELENS_ENV", "development").lower() != "development":
    with st.sidebar:
        st.caption(f"Signed in as {html.escape(str(getattr(st.user, 'email', 'authorized user')))}")
        if st.button("Sign out", width="stretch"):
            st.logout()


@st.cache_resource(max_entries=2, show_spinner=False)
def _train_models_cached(dataset_checksum: str, _frame: pd.DataFrame, _algorithms: dict) -> dict:
    """Share immutable trained models across sessions for identical datasets."""
    return train_models(_frame, algorithms=_algorithms or None)


def train_models_for_dataset(frame: pd.DataFrame, checksum: str, algorithms: dict | None = None) -> dict:
    return _train_models_cached(checksum, frame, algorithms or {})

for key, default in {"data": None, "models": None, "dataset": None, "baseline": None, "scenario": None, "saved_overrides": {}, "current_plan": {}, "override_reason": "No override", "override_revision": 0}.items():
    if key not in st.session_state: st.session_state[key] = default
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if not st.session_state.get("workspace_restore_checked"):
    st.session_state.workspace_restore_checked = True
    try:
        saved = load_workspace()
        if saved:
            st.session_state.data = saved["data"]
            st.session_state.dataset = saved["dataset"]
            st.session_state.baseline = saved["baseline"]
            st.session_state.scenario = saved["scenario"]
            st.session_state.saved_overrides = saved["saved_overrides"]
            st.session_state.current_plan = saved["current_plan"]
            st.session_state.override_reason = saved["override_reason"]
            if st.session_state.dataset.get("schema_mode") == "daily_aggregate_insights":
                st.session_state.models = None
            else:
                with st.spinner("Restoring your saved workspace and rebuilding its models…"):
                    saved_algorithms = st.session_state.dataset.get("algorithms", {})
                    st.session_state.models = train_models_for_dataset(
                        st.session_state.data, st.session_state.dataset["checksum"], saved_algorithms)
                actual_algorithms = {"demand": st.session_state.models["demand_algorithm"], "waste": st.session_state.models["waste_algorithm"]}
                st.session_state.dataset["algorithms"] = actual_algorithms
                st.session_state.dataset["library_versions"] = st.session_state.models["library_versions"]
                st.session_state.dataset["model_version"] = f"demand-{actual_algorithms['demand']}_waste-{actual_algorithms['waste']}-v2-{st.session_state.dataset['checksum'][:8]}"
                save_workspace(st.session_state.data, st.session_state.dataset, st.session_state.baseline,
                               st.session_state.scenario, st.session_state.saved_overrides,
                               st.session_state.current_plan, st.session_state.override_reason)
    except Exception:
        st.session_state.data = None
        st.session_state.models = None
        st.session_state.dataset = None
        st.session_state.workspace_restore_error = True
        st.error("The saved local workspace could not be restored. Its data file may be damaged; clear the local workspace or re-import a valid CSV.")

def persist_workspace():
    if st.session_state.data is None:
        return
    try:
        if get_active_dataset_id() != st.session_state.dataset["id"]:
            st.warning("The active dataset changed in another app session. Reload the workspace before saving this plan.")
            return
        run = st.session_state.baseline
        if run:
            recommendations_frame = recommendations(run["forecast"], run["buffer"], run["increment"], run.get("item_settings"), run.get("risk_thresholds"))
            recommendations_frame["Final portions"] = recommendations_frame.apply(
                lambda row: st.session_state.saved_overrides.get(row["Menu Item"], row["Recommended portions"]), axis=1)
            recommendations_frame["Current planned portions"] = recommendations_frame["Menu Item"].map(st.session_state.current_plan).fillna("Not provided")
            payload = {"run": run, "recommendations": recommendations_frame.to_dict(orient="records"),
                       "overrides": st.session_state.saved_overrides, "current_plan": st.session_state.current_plan,
                       "override_reason": st.session_state.override_reason}
            canonical = save_run(run, payload)
            if canonical["id"] != run["id"]:
                original = get_run_by_request_key(run["request_key"])
                if original:
                    original_payload = original["payload"]
                    st.session_state.baseline = original_payload["run"]
                    st.session_state.saved_overrides = original_payload.get("overrides", {})
                    st.session_state.current_plan = original_payload.get("current_plan", {})
                    st.session_state.override_reason = original_payload.get("override_reason", "No override")
                    st.session_state.scenario = None
        save_workspace(st.session_state.data, st.session_state.dataset, st.session_state.baseline,
                       st.session_state.scenario, st.session_state.saved_overrides,
                       st.session_state.current_plan, st.session_state.override_reason)
    except Exception:
        st.warning("This change is active for the current session, but could not be saved locally. Check available disk space and the configured database path.")
        log_event("workspace_save_failed", session_id=st.session_state.session_id,
                  dataset_id=st.session_state.dataset.get("id"), model_version=st.session_state.dataset.get("model_version"),
                  status="failed", error_code="WL-STORE-503")

def reset_scenario_callback():
    """Clear the what-if before the Streamlit rerun renders the page."""
    st.session_state.scenario = None
    persist_workspace()

def commit_dataset(df, checksum, provenance, warnings):
    trained_models = train_models_for_dataset(df, checksum)
    algorithms = {"demand": trained_models["demand_algorithm"], "waste": trained_models["waste_algorithm"]}
    model_version = f"demand-{algorithms['demand']}_waste-{algorithms['waste']}-v2-{checksum[:8]}"
    dataset_metadata = {"id": checksum[:12], "checksum": checksum, "model_version": model_version,
                        "algorithms": algorithms, "library_versions": trained_models["library_versions"],
                        "rows": len(df), "provenance": provenance, "warnings": warnings,
                        "imported_at": dt.datetime.now().astimezone().isoformat()}
    # Commit durable data first so a storage error cannot silently replace the active session dataset.
    save_workspace(df.reset_index(drop=True), dataset_metadata, None, None, {}, {}, "No override")
    st.session_state.data = df.reset_index(drop=True)
    st.session_state.dataset = dataset_metadata
    st.session_state.models = trained_models
    st.session_state.baseline = None
    st.session_state.scenario = None
    st.session_state.saved_overrides = {}
    st.session_state.current_plan = {}
    st.session_state.override_reason = "No override"


def commit_aggregate_dataset(df, checksum, provenance, warnings):
    metadata = {"id": checksum[:12], "checksum": checksum, "model_version": "aggregate-insights-v1",
                "schema_mode": "daily_aggregate_insights", "rows": len(df), "provenance": provenance,
                "warnings": warnings, "imported_at": dt.datetime.now().astimezone().isoformat()}
    save_workspace(df.reset_index(drop=True), metadata, None, None, {}, {}, "No override")
    st.session_state.data = df.reset_index(drop=True)
    st.session_state.dataset = metadata
    st.session_state.models = None
    st.session_state.baseline = None
    st.session_state.scenario = None
    st.session_state.saved_overrides = {}
    st.session_state.current_plan = {}
    st.session_state.override_reason = "No override"


def generate_or_restore_plan(inputs, buffer, increment, item_settings, risk_thresholds):
    started = time.perf_counter()
    key = request_key(metadata["id"], metadata["model_version"], inputs, buffer, increment, item_settings, risk_thresholds)
    if get_active_dataset_id() != metadata["id"]:
        st.error("The dataset changed in another app session. Reload the workspace before generating a plan.")
        return False
    log_event("plan_generation_started", session_id=st.session_state.session_id,
              dataset_id=metadata["id"], model_version=metadata["model_version"], status="started")
    previous = get_run_by_request_key(key)
    if previous:
        payload = previous["payload"]
        st.session_state.baseline = payload["run"]
        st.session_state.saved_overrides = payload.get("overrides", {})
        st.session_state.current_plan = payload.get("current_plan", {})
        st.session_state.override_reason = payload.get("override_reason", "No override")
        st.session_state.scenario = None
        st.info(f"Recovered the original result for this identical request (run {previous['run_id']}).")
        log_event("plan_generation_completed", session_id=st.session_state.session_id,
                  dataset_id=metadata["id"], model_version=metadata["model_version"],
                  duration_ms=(time.perf_counter() - started) * 1000, status="recovered", run_id=previous["run_id"])
    else:
        try:
            with st.spinner("Generating and explaining your forecast…"):
                result = forecast(models, df, inputs)
        except Exception:
            identifier = trace_id()
            log_event("plan_generation_failed", session_id=st.session_state.session_id,
                      dataset_id=metadata["id"], model_version=metadata["model_version"],
                      duration_ms=(time.perf_counter() - started) * 1000, status="failed",
                      error_code="WL-MODEL-503", trace=identifier)
            st.error(f"Prediction is unavailable. Your previous successful plan remains available; retry shortly. Reference: {identifier} (WL-MODEL-503).")
            return False
        if "error" in result:
            identifier = trace_id()
            log_event("plan_generation_failed", session_id=st.session_state.session_id,
                      dataset_id=metadata["id"], model_version=metadata["model_version"],
                      duration_ms=(time.perf_counter() - started) * 1000, status="failed",
                      error_code="WL-INPUT-422", trace=identifier)
            st.error(f"Prediction is unavailable. Check the selected meal and menu, then retry. Reference: {identifier} (WL-INPUT-422).")
            return False
        run = {"id": uuid.uuid4().hex[:8], "request_key": key, "generated_at": dt.datetime.now().astimezone().isoformat(),
               "dataset_id": metadata["id"], "model_version": metadata["model_version"], "inputs": inputs,
               "forecast": result, "buffer": buffer, "increment": int(increment),
               "item_settings": item_settings, "risk_thresholds": risk_thresholds}
        log_event("model_postprocessing", session_id=st.session_state.session_id,
                  dataset_id=metadata["id"], model_version=metadata["model_version"], run_id=run["id"],
                  status="warning" if result.get("warnings") else "success",
                  model_outputs={"raw_demand": result["raw_demand"],
                                 "raw_waste_predictions_kg": [item["raw_waste_kg"] for item in result["items"]],
                                 "nonnegative_clipped": any("clipped to zero" in warning for warning in result.get("warnings", []))})
        st.session_state.baseline = run
        st.session_state.scenario = None
        st.session_state.saved_overrides = {}
        st.session_state.current_plan = {}
        st.session_state.override_reason = "No override"
        log_event("plan_generation_completed", session_id=st.session_state.session_id,
                  dataset_id=metadata["id"], model_version=metadata["model_version"],
                  duration_ms=(time.perf_counter() - started) * 1000, run_id=run["id"])
    persist_workspace()
    return True


def configure_pdf_font(pdf):
    candidates = [
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdf.add_font("WasteLensUnicode", "", str(regular), uni=True)
            pdf.add_font("WasteLensUnicode", "B", str(bold), uni=True)
            return "WasteLensUnicode", True
    return "Arial", False


def csv_download_bytes(frame: pd.DataFrame) -> bytes:
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]):
        safe[column] = safe[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    return safe.to_csv(index=False).encode("utf-8-sig")

with st.sidebar:
    st.header("Historical data")
    template = pd.DataFrame(columns=REQUIRED).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download CSV template", template, "wastelens_template.csv", "text/csv", width="stretch")
    st.caption("Required columns: " + ", ".join(f"`{column}`" for column in REQUIRED))
    with st.form("upload_form", clear_on_submit=False):
        upload = st.file_uploader("Upload a UTF-8 CSV (up to 10 MB)", type=["csv"])
        replace_ok = st.checkbox("I understand that importing replaces the current dataset and forecasts.", value=st.session_state.data is None)
        submitted = st.form_submit_button("Validate and import", type="primary", width="stretch")
    if submitted:
        if upload is None:
            st.error("Choose a CSV file first.")
        elif not replace_ok:
            st.error("Confirm the dataset replacement to continue.")
        else:
            raw_upload = upload.getvalue()
            validation_started = time.perf_counter()
            log_event("dataset_upload_started", session_id=st.session_state.session_id)
            with st.status("Uploading → Validating → Ready", expanded=True) as upload_status:
                upload_status.write("Uploading: file received; checking the 10 MB limit.")
                upload_status.write("Validating: checking columns, rows, units, duplicates, and usable history.")
                validation = validate_csv(raw_upload)
                upload_status.update(label="Ready" if not validation.errors else "Validation needs attention",
                                     state="complete" if not validation.errors else "error", expanded=False)
            st.session_state.pending_import = {"frame": validation.frame, "errors": validation.errors, "warnings": validation.warnings,
                                               "rejected": validation.rejected, "valid_rows": validation.valid_rows,
                                               "total_rows": validation.total_rows, "checksum": validation.checksum,
                                               "warning_rows": validation.warning_rows,
                                               "mappings": validation.column_mappings or {}, "raw": raw_upload}
            log_event("dataset_validation_completed", session_id=st.session_state.session_id,
                      dataset_id=validation.checksum[:12] if validation.checksum else None,
                      duration_ms=(time.perf_counter() - validation_started) * 1000,
                      status="blocked" if validation.errors else "success",
                      error_code="WL-DATA-422" if validation.errors else None)
            st.rerun()
    pending = st.session_state.get("pending_import")
    if pending:
        st.markdown("#### Import preview")
        for message in pending["errors"]: st.error(message)
        for message in pending["warnings"]: st.warning(message)
        if pending.get("mappings"):
            st.write("**Column mapping preview**", pending["mappings"])
        rejected_rows = int(len(pending["rejected"]))
        st.caption(f"{pending['valid_rows']:,} valid · {pending.get('warning_rows', 0):,} valid rows with warnings · {rejected_rows:,} rejected of {pending['total_rows']:,} total rows")
        if pending["frame"] is not None:
            st.dataframe(pending["frame"].head(20), hide_index=True, width="stretch")
        if len(pending["rejected"]):
            st.download_button("Download rejected rows", csv_download_bytes(pending["rejected"]), "wastelens_rejected_rows.csv", "text/csv")
        if any("duplicate date/meal/item keys" in error.lower() for error in pending["errors"]):
            if st.button("Keep last duplicate row and revalidate"):
                with st.spinner("Applying the confirmed keep-last rule…"):
                    validation = validate_csv(pending["raw"], keep_last_duplicates=True)
                st.session_state.pending_import = {"frame": validation.frame, "errors": validation.errors, "warnings": validation.warnings,
                                                   "rejected": validation.rejected, "valid_rows": validation.valid_rows,
                                                   "total_rows": validation.total_rows, "checksum": validation.checksum,
                                                   "warning_rows": validation.warning_rows,
                                                   "mappings": validation.column_mappings or {}, "raw": pending["raw"]}
                st.rerun()
        if pending["frame"] is not None and not pending["errors"]:
            if st.button("Confirm import and train", type="primary", width="stretch"):
                try:
                    with st.spinner("Training and evaluating on your data…"):
                        commit_dataset(pending["frame"], pending["checksum"], "User upload", pending["warnings"])
                    st.session_state.pending_import = None
                    log_event("dataset_import_committed", session_id=st.session_state.session_id,
                              dataset_id=st.session_state.dataset["id"], model_version=st.session_state.dataset["model_version"])
                except Exception:
                    identifier = trace_id()
                    log_event("dataset_import_failed", session_id=st.session_state.session_id,
                              dataset_id=pending.get("checksum", "")[:12], status="failed",
                              error_code="WL-IMPORT-503", trace=identifier)
                    st.error(f"The import could not be completed. Your previous active dataset remains available; check the data or writable local storage, then retry. Reference: {identifier} (WL-IMPORT-503).")
        if st.button("Cancel import", width="stretch"):
            st.session_state.pending_import = None
            st.rerun()
    st.divider()
    with st.expander("Import daily aggregate data (insights only)"):
        st.caption("For datasets with daily totals but no per-item prepared/served counts. WasteLens will show historical trends only and will not generate kitchen recommendations.")
        aggregate_upload = st.file_uploader("Choose aggregate CSV or XLSX (up to 10 MB)", type=["csv", "xlsx", "xlsm"], key="aggregate_upload")
        if aggregate_upload is not None:
            aggregate_raw = aggregate_upload.getvalue()
            try:
                aggregate_source = read_aggregate_file(aggregate_raw, aggregate_upload.name)
                guessed = suggest_aggregate_columns(aggregate_source.columns)
                aggregate_columns = ["—"] + aggregate_source.columns.tolist()
                def select_aggregate_column(label, field, required=False):
                    guess = guessed.get(field)
                    index = aggregate_columns.index(guess) if guess in aggregate_columns else 0
                    value = st.selectbox(label + (" *" if required else " (optional)"), aggregate_columns,
                                         index=index, key=f"agg_map_{field}")
                    return None if value == "—" else value
                date_column = select_aggregate_column("Service date", "date", True)
                diners_column = select_aggregate_column("Diners / customers", "diners")
                plate_column = select_aggregate_column("Plate waste (kg)", "plate_waste")
                kitchen_column = select_aggregate_column("Kitchen and serving waste (kg)", "kitchen_waste")
                menu_column = select_aggregate_column("Menu identifier", "menu")
                replace_aggregate = st.checkbox("I understand this replaces the current dataset and forecasts.", key="aggregate_replace")
                if st.button("Validate and import aggregate insights", type="primary", width="stretch"):
                    if not replace_aggregate:
                        st.error("Confirm the dataset replacement to continue.")
                    else:
                        mapping = {"date": date_column, "diners": diners_column, "plate_waste": plate_column,
                                   "kitchen_waste": kitchen_column, "menu": menu_column}
                        try:
                            aggregate_frame, aggregate_warnings = normalize_aggregate(aggregate_source, mapping)
                            commit_aggregate_dataset(aggregate_frame, aggregate_checksum(aggregate_raw),
                                                     "User upload · daily aggregate", aggregate_warnings)
                            st.success("Daily aggregate insights imported. Item-level planning is disabled for this schema.")
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))
            except ValueError as error:
                st.error(str(error))
    st.divider()
    st.caption("Demo data is synthetic and labeled as such.")
    if st.button("Load synthetic demo", width="stretch"):
        try:
            raw = (Path(__file__).resolve().parent / "data" / "synthetic_historical_data.csv").read_bytes()
            result = validate_csv(raw)
            if result.errors: st.error("Demo data did not pass validation: " + "; ".join(result.errors))
            else:
                with st.spinner("Training and evaluating demo models…"):
                    commit_dataset(result.frame, result.checksum, "Synthetic demo", result.warnings)
        except Exception: st.error("Could not load the demo dataset. Check that the demo CSV is present and readable.")
    if st.session_state.data is not None:
        if st.button("Clear data and forecasts", width="stretch"):
            st.session_state.clear_data_pending = True
        if st.session_state.get("clear_data_pending"):
            with st.form("clear_data_confirmation"):
                confirm_clear = st.checkbox("Confirm clearing the active dataset and all forecasts.")
                clear_submitted = st.form_submit_button("Confirm clear")
            if clear_submitted:
                if confirm_clear:
                    try:
                        if get_active_dataset_id() != st.session_state.dataset["id"]:
                            st.error("The dataset changed in another app session. Reload it before clearing data.")
                            st.stop()
                        clear_workspace_and_runs()
                    except Exception:
                        st.error("The local workspace could not be cleared. Close other WasteLens instances and retry.")
                        st.stop()
                    for key in ["data", "models", "dataset", "baseline", "scenario"]: st.session_state[key] = None
                    st.session_state.saved_overrides = {}
                    st.session_state.current_plan = {}
                    st.session_state.clear_data_pending = False
                    st.rerun()
                st.error("Check the confirmation box to clear the active dataset.")
    elif st.session_state.get("workspace_restore_error"):
        if st.button("Clear damaged local workspace"):
            clear_workspace_and_runs()
            st.session_state.workspace_restore_error = False
            st.rerun()

if st.session_state.data is None:
    render_empty_state()
    st.info("Upload historical kitchen data or load the labeled synthetic demo dataset to begin.")
    st.stop()

df, models, metadata = st.session_state.data, st.session_state.models, st.session_state.dataset
if get_active_dataset_id() != metadata["id"]:
    st.error("This workspace changed in another browser session. Reload to use the current dataset and avoid overwriting its saved plans.")
    if st.button("Reload active workspace"):
        for key in ["data", "models", "dataset", "baseline", "scenario"]: st.session_state[key] = None
        st.session_state.saved_overrides = {}
        st.session_state.current_plan = {}
        st.session_state.workspace_restore_checked = False
        st.rerun()
    st.stop()
if metadata.get("schema_mode") == "daily_aggregate_insights":
    st.info("Insights-only dataset: daily aggregate measurements do not include per-item meals prepared and served, so WasteLens does not generate item forecasts or recommendations.")
    for warning in metadata.get("warnings", []):
        st.warning(warning)
    daily = df.groupby("date", as_index=False).agg(
        diners=("diners", lambda values: values.sum(min_count=1)),
        plate_waste_kg=("plate_waste_kg", lambda values: values.sum(min_count=1)),
        kitchen_serving_waste_kg=("kitchen_serving_waste_kg", lambda values: values.sum(min_count=1)),
        total_measured_waste_kg=("total_measured_waste_kg", lambda values: values.sum(min_count=1)),
    ).sort_values("date")
    total_waste = df["total_measured_waste_kg"].sum(min_count=1)
    diners = df["diners"].sum(min_count=1)
    per_diner = total_waste / diners if pd.notna(total_waste) and pd.notna(diners) and diners > 0 else np.nan
    a, b, c, d = st.columns(4)
    a.metric("Days recorded", f"{df['date'].nunique():,}")
    b.metric("Measured waste", f"{total_waste:,.1f} kg" if pd.notna(total_waste) else "Not available")
    c.metric("Diners recorded", f"{diners:,.0f}" if pd.notna(diners) else "Not available")
    d.metric("Waste per diner", f"{per_diner * 1000:,.0f} g" if pd.notna(per_diner) else "Not available")
    st.caption(f"Dataset {metadata['id']} · {metadata['provenance']} · {df['date'].min().date()} to {df['date'].max().date()}")
    st.subheader("Daily measured waste")
    chart = daily.set_index("date")[["plate_waste_kg", "kitchen_serving_waste_kg"]].rename(
        columns={"plate_waste_kg": "Plate waste (kg)", "kitchen_serving_waste_kg": "Kitchen and serving waste (kg)"})
    # Preserve missing days as gaps and keep this view off Vega, whose renderer
    # caused blank-screen failures in earlier Streamlit/browser combinations.
    chart = chart.reindex(pd.date_range(chart.index.min(), chart.index.max(), freq="D"))
    figure, axis = plt.subplots(figsize=(10, 4))
    for column in chart.columns:
        axis.plot(chart.index, chart[column], marker="o", linewidth=1.5, markersize=3, label=column)
    axis.set_ylabel("Measured waste (kg)")
    axis.set_xlabel("Service date")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=.2)
    figure.autofmt_xdate()
    figure.tight_layout()
    st.pyplot(figure, width="stretch", clear_figure=True)
    st.caption("Each plotted point reflects recorded measurements only. Missing dates are not interpolated. Daily waste per diner uses measured total kilograms divided by the mapped diner count.")
    st.dataframe(df.sort_values("date", ascending=False), hide_index=True, width="stretch")
    st.stop()
if metadata["provenance"] == "Synthetic demo": st.warning("Synthetic demo data is illustrative. Do not use its forecast as an operational recommendation.", icon="⚠️")
for quality_warning in metadata["warnings"]:
    st.warning(quality_warning)
with st.expander("Dataset quality and model evaluation", expanded=False):
    st.write(f"**Dataset:** `{metadata['id']}` · {metadata['rows']:,} rows · {metadata['provenance']} · {df.date.min().date()} to {df.date.max().date()}")
    st.write(f"**SHA-256:** `{metadata['checksum']}`")
    if metadata["warnings"]:
        for warning in metadata["warnings"]: st.warning(warning)
    for target, label in [("demand_metrics", "Demand"), ("waste_metrics", "Waste")]:
        m = models[target]
        wape = f"{m['wape']:.1%}" if m["wape"] is not None else "n/a (zero actual total)"
        st.write(f"**{label} holdout ({m['algorithm']})** — MAE {m['mae']:.2f}; RMSE {m['rmse']:.2f}; R² {m['r2']:.2f}; WAPE {wape}; {m['baseline_method']} MAE {m['baseline_mae']:.2f} (n={m['rows']})")
        st.caption("Candidate comparison: " + "; ".join(f"{name}: MAE {values['mae']:.2f}, stability SD {values['stability_mae_std']:.2f}, fit {values['fit_ms']:.0f} ms, inference {values['inference_ms']:.2f} ms" for name, values in m["model_comparison"].items()))
    st.markdown("**Holdout performance by data slice**")
    for target, label in [("demand_metrics", "Demand"), ("waste_metrics", "Waste")]:
        st.markdown(f"_{label}_")
        for dimension, rows in models[target]["slices"].items():
            if rows:
                st.write(f"{dimension}: " + ", ".join(f"{r['slice']} (MAE {r['mae']:.2f}, n={r['rows']})" for r in rows))
    if models["quality"] == "weak-baseline": st.error("At least one model does not beat its simple median baseline on the chronological holdout. Treat forecasts as low confidence.")
    else: st.success("Both models beat their grouped historical-median baseline on the chronological holdout. This is an offline check, not a guarantee of future accuracy.")

tabs = st.tabs(["Plan tomorrow", "What-if", "Historical insights", "Saved plans", "Data & model"])
all_meals = sorted(df.meal_type.dropna().astype(str).unique().tolist())
all_items = sorted(df.menu_item.dropna().astype(str).unique().tolist())

with tabs[0]:
    st.subheader("Planning inputs")
    with st.form("plan_form"):
        a, b, c = st.columns(3)
        with a:
            target_date = st.date_input("Target date", value=dt.date.today() + dt.timedelta(days=1))
            meal = st.selectbox("Meal period", all_meals)
        with b:
            latest = df[df.meal_type.astype(str) == meal].sort_values("date").groupby("date").expected_customers.median()
            customer_default = int(latest.iloc[-1]) if len(latest) and pd.notna(latest.iloc[-1]) else int(df.meals_served.median())
            customers = st.number_input("Expected customers", min_value=0, max_value=MAX_EXPECTED_CUSTOMERS, value=max(0, min(MAX_EXPECTED_CUSTOMERS, customer_default)), step=1)
            temperature = st.number_input("Temperature (°C)", value=float(pd.to_numeric(df.temperature, errors="coerce").median()) if df.temperature.notna().any() else 25.0)
        with c:
            holiday = st.checkbox("Public holiday")
            event = st.checkbox("Special event")
        items = st.multiselect("Menu items", all_items, default=all_items)
        buffer_pct = st.number_input("Safety buffer (%)", min_value=0.0, max_value=50.0, value=DEFAULT_BUFFER * 100, step=1.0)
        increment = st.number_input("Default production increment (portions)", min_value=1, max_value=1000, value=DEFAULT_PRODUCTION_INCREMENT, step=1)
        moderate_risk_pct = st.number_input("Moderate waste-risk threshold (%)", min_value=0.0, max_value=99.0, value=DEFAULT_MODERATE_RISK_RATE * 100, step=1.0)
        high_risk_pct = st.number_input("High waste-risk threshold (%)", min_value=1.0, max_value=100.0, value=DEFAULT_HIGH_RISK_RATE * 100, step=1.0)
        settings_seed = pd.DataFrame({"Menu Item": items, "Increment (portions)": int(increment), "Capacity (portions; 0 = unlimited)": 0})
        edited_settings = st.data_editor(settings_seed, hide_index=True, num_rows="fixed", width="stretch",
                                         key=f"item_settings_{metadata['id']}_{'|'.join(sorted(items))}",
                                         disabled=["Menu Item"],
                                         column_config={"Increment (portions)": st.column_config.NumberColumn(min_value=1, max_value=1000, step=1),
                                                        "Capacity (portions; 0 = unlimited)": st.column_config.NumberColumn(min_value=0, step=1)})
        submitted_plan = st.form_submit_button("Generate plan", type="primary", width="stretch")
    item_settings = {
        str(row["Menu Item"]): {"increment": int(row["Increment (portions)"]),
                                "capacity": int(row["Capacity (portions; 0 = unlimited)"]) or None}
        for _, row in edited_settings.iterrows()
    }
    risk_thresholds = {"moderate": moderate_risk_pct / 100, "high": high_risk_pct / 100}
    if submitted_plan:
        if customers == 0:
            st.session_state.confirm_zero = True
        else: st.session_state.confirm_zero = False
        if not items: st.error("Select at least one menu item.")
        elif customers == 0: st.warning("Zero customers may mean a closure. Confirm this plan below before generating.")
        else:
            inputs = {"target_date": target_date, "meal_type": meal, "expected_customers": int(customers), "temperature": float(temperature), "is_holiday": holiday, "has_event": event, "menu_items": items}
            selected_settings = {item: item_settings[item] for item in items}
            if moderate_risk_pct >= high_risk_pct:
                st.error("The high-risk threshold must be greater than the moderate-risk threshold.")
            elif generate_or_restore_plan(inputs, buffer_pct / 100, int(increment), selected_settings, risk_thresholds):
                st.session_state.confirm_zero = False
    if st.session_state.get("confirm_zero"):
        with st.form("closure_confirm"):
            confirm = st.checkbox("Confirm that the kitchen is closed or expects zero customers.")
            go = st.form_submit_button("Confirm and generate zero-customer plan")
        if go and confirm:
            inputs = {"target_date": target_date, "meal_type": meal, "expected_customers": 0, "temperature": float(temperature), "is_holiday": holiday, "has_event": event, "menu_items": items}
            if not items: st.warning("Select at least one menu item.")
            elif moderate_risk_pct >= high_risk_pct:
                st.error("The high-risk threshold must be greater than the moderate-risk threshold.")
            elif generate_or_restore_plan(inputs, buffer_pct / 100, int(increment), {item: item_settings[item] for item in items}, risk_thresholds):
                st.session_state.confirm_zero = False
    run = st.session_state.baseline
    if run:
        if run["dataset_id"] != metadata["id"] or run["model_version"] != metadata["model_version"]: st.error("This plan is stale because the dataset or model version changed. Generate it again.")
        else:
            f = run["forecast"]
            days_ahead = (run["inputs"]["target_date"] - dt.date.today()).days
            if days_ahead > 30: st.warning("This date is more than 30 days away; forecast reliability is lower.")
            if days_ahead < 0: st.info("Historical test forecast.")
            if models["quality"] == "weak-baseline": st.warning("Low confidence: holdout performance did not beat the simple baseline for both targets.")
            if f["excluded_items"]: st.warning("No matching historical rows for: " + ", ".join(f["excluded_items"]))
            for warning in f.get("warnings", []): st.warning(warning)
            confidence = "Low" if models["quality"] == "weak-baseline" else "Degraded" if metadata["warnings"] or f.get("warnings") or days_ahead > 30 or days_ahead < 0 else "Standard"
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Expected customers", f"{run['inputs']['expected_customers']:,}")
            k2.metric("Predicted demand", f"{f['demand']:.0f} meals")
            k3.metric("Expected item waste", f"{sum(i['waste_kg'] for i in f['items']):.2f} kg")
            k4.metric("Forecast status", confidence)
            st.caption("Generated " + run["generated_at"][:19].replace("T", " "))
            st.caption("Forecast status is a qualitative quality signal based on holdout performance, history depth, input range, and date horizon—not a calibrated probability.")
            recs = recommendations(f, run["buffer"], run["increment"], run.get("item_settings"), run.get("risk_thresholds"))
            recs.insert(0, "Run ID", run["id"])
            recs.insert(1, "Dataset ID", run["dataset_id"])
            recs.insert(2, "Model version", run["model_version"])
            # Keep final quantities and their provenance visible and in the export.
            recs["Recommended portions"] = recs["Recommended portions"].astype(str)
            recs["Final portions"] = recs.apply(lambda row: st.session_state.saved_overrides.get(row["Menu Item"], row["Recommended portions"]), axis=1)
            recs["Current planned portions"] = recs["Menu Item"].map(st.session_state.current_plan).fillna("Not provided")
            current_numbers = pd.to_numeric(recs["Current planned portions"], errors="coerce")
            final_numbers = pd.to_numeric(recs["Final portions"], errors="coerce")
            comparable = current_numbers.notna() & final_numbers.notna()
            recs["Estimated waste avoided (portions)"] = pd.Series("Not available", index=recs.index, dtype=object)
            recs.loc[comparable, "Estimated waste avoided (portions)"] = np.maximum(0, current_numbers[comparable] - final_numbers[comparable]).astype(int)
            st.subheader("Item recommendations")
            st.caption("Item portions scale from each item's historical served portions relative to the highest-served item for that meal and date. Waste risk uses `portion_mass_kg`; without it, risk is unavailable because kilograms cannot be compared directly with portions.")
            recs = recs.sort_values(["Predicted waste (kg)", "Menu Item"], ascending=[False, True])
            item_query = st.text_input("Search recommendations", placeholder="Find a menu item")
            display_recs = recs[recs["Menu Item"].str.contains(item_query, case=False, na=False)] if item_query else recs.head(10)
            if len(display_recs) == 0:
                st.info("No menu items match this search.")
            else:
                for _, recommendation in display_recs.iterrows():
                    with st.container(border=True):
                        item_name, plan_value, waste_value = st.columns([2, 1, 1])
                        item_name.markdown(f"**{html.escape(str(recommendation['Menu Item']))}** · {recommendation['Waste risk']} waste risk")
                        item_name.caption(f"Recommendation {recommendation['Final portions']} portions; predicted waste {recommendation['Predicted waste (kg)']:.2f} kg; waste risk {recommendation['Waste risk']}.")
                        item_name.caption(f"Portion mass: {recommendation['Portion mass (kg)']}")
                        item_name.caption("Risk policy: " + recommendation["Risk policy"])
                        item_name.caption(f"Capacity: {recommendation['Capacity portions']} portions · unmet demand risk: {recommendation['Unmet demand risk']}")
                        if recommendation["Unmet demand risk"] == "High":
                            item_name.error(f"Capacity is below forecast demand. Uncapped plan: {recommendation['Uncapped recommendation']} portions; capped recommendation: {recommendation['Recommended portions']}.")
                        if recommendation["Waste drivers"]:
                            item_name.caption("Waste model drivers: " + recommendation["Waste drivers"])
                        current_value = pd.to_numeric(recommendation["Current planned portions"], errors="coerce")
                        if pd.notna(current_value):
                            item_name.caption(f"Current plan: {int(current_value)} · estimated waste avoided: {recommendation['Estimated waste avoided (portions)']} portions (plan difference estimate, not measured waste)")
                        forecast_value = pd.to_numeric(recommendation["Forecast portions"], errors="coerce")
                        final_value = pd.to_numeric(recommendation["Final portions"], errors="coerce")
                        if pd.notna(forecast_value) and pd.notna(final_value) and final_value < forecast_value:
                            item_name.warning(f"Final quantity is {int(forecast_value - final_value)} portions below the demand forecast.")
                        plan_value.metric("Recommended", recommendation["Recommended portions"], delta=f"Final {recommendation['Final portions']}", delta_color="off")
                        waste_value.metric("Predicted waste", f"{recommendation['Predicted waste (kg)']:.2f} kg")
                if not item_query and len(recs) > 10:
                    st.caption(f"Showing the 10 highest-waste items of {len(recs)}. Search to find another item; exports contain all items.")
            st.markdown("#### Adjust final quantities")
            with st.form("overrides"):
                changed = dict(st.session_state.saved_overrides)
                current_changed = dict(st.session_state.current_plan)
                cols = st.columns(min(3, max(1, len(display_recs))))
                for idx, row in display_recs.iterrows():
                    rec = row["Recommended portions"]
                    if rec.isdigit():
                        with cols[idx % len(cols)]:
                            has_current = st.checkbox(f"{row['Menu Item']} has a current plan", value=row["Menu Item"] in st.session_state.current_plan, key=f"has_current_{run['id']}_{st.session_state.override_revision}_{idx}")
                            current_value = st.number_input(f"{row['Menu Item']} current portions (if checked)", min_value=0, value=int(st.session_state.current_plan.get(row["Menu Item"], 0)), key=f"current_{run['id']}_{st.session_state.override_revision}_{idx}")
                            if has_current: current_changed[row["Menu Item"]] = current_value
                            else: current_changed.pop(row["Menu Item"], None)
                            changed[row["Menu Item"]] = st.number_input(f"{row['Menu Item']} final portions", min_value=0, value=int(st.session_state.saved_overrides.get(row["Menu Item"], rec)), key=f"ov_{run['id']}_{st.session_state.override_revision}_{idx}")
                reason = st.selectbox("Reason for quantity overrides", ["No override", "Chef judgment", "Updated attendance", "Ingredient availability", "Other"], index=["No override", "Chef judgment", "Updated attendance", "Ingredient availability", "Other"].index(st.session_state.override_reason), key=f"reason_{run['id']}_{st.session_state.override_revision}")
                save = st.form_submit_button("Save overrides")
            if save:
                st.session_state.saved_overrides = changed
                st.session_state.current_plan = current_changed
                st.session_state.override_reason = reason
                persist_workspace()
                log_event("recommendation_overridden", session_id=st.session_state.session_id,
                          dataset_id=metadata["id"], model_version=metadata["model_version"],
                          status="overridden" if changed else "accepted", run_id=run["id"])
                st.rerun()
            if st.session_state.saved_overrides and st.button("Reset all quantities to recommendations"):
                st.session_state.saved_overrides = {}
                st.session_state.override_reason = "No override"
                st.session_state.override_revision += 1
                persist_workspace()
                st.rerun()
            st.caption(f"Dataset {run['dataset_id']} · model {run['model_version']} · run {run['id']} · safety buffer {run['buffer']:.0%} · portions rounded up to increments of {run['increment']}.")
            show_explanation = st.checkbox("Show demand forecast explanation", key=f"show_explanation_{run['id']}")
            if show_explanation and st.session_state.get("explanation_logged_run") != run["id"]:
                log_event("explanation_viewed", session_id=st.session_state.session_id,
                          dataset_id=metadata["id"], model_version=metadata["model_version"], run_id=run["id"])
                st.session_state.explanation_logged_run = run["id"]
            if show_explanation:
                explanation = f.get("explanation", {})
                st.caption(explanation.get("kind", "Explanation unavailable").replace("_", " "))
                for driver in explanation.get("drivers", []):
                    name = driver["feature"].replace("_", " ").replace("meal type ", "meal: ").title()
                    if explanation.get("kind") == "local SHAP":
                        st.write(f"{name}: {driver['impact']:+.1f} meals of model contribution")
                    else:
                        st.write(f"{name}: {driver['impact']:.1%} global importance")
                st.caption("Demand drivers show model influence, not cause. Global importance is a fallback when local explanation is unavailable; item cards also show each waste forecast's drivers.")
            export_frame = recs.copy()
            export_frame["Override reason"] = st.session_state.override_reason
            export = csv_download_bytes(export_frame)
            if st.download_button("Download decision plan (CSV)", export, f"wastelens_{run['inputs']['target_date']}_{run['id']}.csv", "text/csv", key=f"plan_csv_{run['id']}"):
                log_event("report_downloaded", session_id=st.session_state.session_id,
                          dataset_id=metadata["id"], model_version=metadata["model_version"], run_id=run["id"])
            try:
                from fpdf import FPDF
                pdf = FPDF(); pdf.add_page()
                pdf_font, unicode_pdf = configure_pdf_font(pdf)
                pdf.set_font(pdf_font, "B", 16); pdf.cell(0, 10, "WasteLens Decision Plan", ln=True)
                pdf.set_font(pdf_font, size=10)
                report_lines = [f"Run: {run['id']}   Generated: {run['generated_at']}", f"Dataset: {run['dataset_id']} ({metadata['provenance']})   Model: {run['model_version']}", f"Date: {run['inputs']['target_date']}   Meal: {run['inputs']['meal_type']}", f"Forecast mode: {'Historical test' if days_ahead < 0 else 'Planning'}", f"Expected customers: {run['inputs']['expected_customers']}   Temperature: {run['inputs']['temperature']:.1f} C", f"Holiday: {run['inputs']['is_holiday']}   Event: {run['inputs']['has_event']}", f"Predicted demand: {f['demand']:.0f} meals", f"Predicted waste: {sum(i['waste_kg'] for i in f['items']):.2f} kg", f"Forecast status: {confidence} (qualitative, not a probability)", f"Model status: {models['quality']} (holdout metrics shown in app)"]
                if days_ahead > 30: report_lines.append("Warning: Planning date is more than 30 days away; forecast reliability is lower.")
                report_lines.extend("Warning: " + warning for warning in f.get("warnings", []))
                report_lines.extend("Data quality: " + warning for warning in metadata["warnings"])
                if any(i["portion_mass_kg"] is None for i in f["items"]): report_lines.append("Waste risk is unavailable for one or more items because portion mass was not supplied.")
                for line in report_lines:
                    pdf.multi_cell(0, 7, line)
                pdf.ln(3)
                for _, row in recs.iterrows(): pdf.multi_cell(0, 7, f"{row['Menu Item']}: forecast {row['Forecast portions']}, recommended {row['Recommended portions']}, capacity {row['Capacity portions']}, unmet demand risk {row['Unmet demand risk']}, current {row['Current planned portions']}, final {row['Final portions']}, estimated waste avoided {row['Estimated waste avoided (portions)']} portions, predicted waste {row['Predicted waste (kg)']} kg, risk {row['Waste risk']}")
                if st.session_state.override_reason != "No override": pdf.multi_cell(0, 7, f"Override reason: {st.session_state.override_reason}")
                if not unicode_pdf: st.caption("Unicode fonts are unavailable; some non-Latin characters may be omitted from the PDF. CSV preserves UTF-8.")
                if st.download_button("Download decision report (PDF)", bytes(pdf.output(dest="S").encode("latin-1")), f"wastelens_{run['inputs']['target_date']}_{run['id']}.pdf", "application/pdf", key=f"plan_pdf_{run['id']}"):
                    log_event("report_downloaded", session_id=st.session_state.session_id,
                              dataset_id=metadata["id"], model_version=metadata["model_version"], run_id=run["id"])
            except Exception:
                st.info("PDF generation is unavailable in this environment; the complete CSV and on-screen plan remain available.")

with tabs[1]:
    st.subheader("What-if scenario")
    base = st.session_state.baseline
    if base is None: st.info("Generate a plan first to compare a scenario.")
    elif base["dataset_id"] != metadata["id"] or base["model_version"] != metadata["model_version"]: st.error("The baseline belongs to a replaced dataset or model version; regenerate it before comparing.")
    else:
        st.caption(f"Baseline run {base['id']} stays unchanged while you explore scenarios.")
        with st.form("scenario_form"):
            si = base["inputs"]
            sc1, sc2, sc3 = st.columns(3)
            with sc1: sc_customers = st.number_input("Scenario customers", 0, 100000, si["expected_customers"])
            with sc2: sc_temp = st.number_input("Scenario temperature (°C)", value=float(si["temperature"]))
            with sc3: sc_event = st.checkbox("Special event", value=si["has_event"])
            sc_items = st.multiselect("Scenario menu", all_items, default=si["menu_items"])
            unchanged = (sc_customers == si["expected_customers"] and sc_temp == si["temperature"] and
                         sc_event == si["has_event"] and sorted(sc_items) == sorted(si["menu_items"]))
            apply = st.form_submit_button("Apply scenario", type="primary")
        if apply:
            if sc_customers == 0:
                st.session_state.pending_zero_scenario = {"customers": int(sc_customers), "temperature": float(sc_temp), "event": sc_event, "items": sc_items}
                st.warning("Zero customers require closure confirmation before applying this scenario.")
            elif not sc_items: st.warning("Select at least one menu item.")
            elif sc_customers == si["expected_customers"] and sc_temp == si["temperature"] and sc_event == si["has_event"] and sc_items == si["menu_items"]: st.warning("Change at least one input before applying a scenario.")
            else:
                changed_inputs = dict(si, expected_customers=int(sc_customers), temperature=float(sc_temp), has_event=sc_event, menu_items=sc_items)
                try:
                    sf = forecast(models, df, changed_inputs)
                    if "error" in sf:
                        st.error(sf["error"])
                    else:
                        st.session_state.scenario = {"baseline_run_id": base["id"], "inputs": changed_inputs, "forecast": sf, "generated_at": dt.datetime.now().astimezone().isoformat()}
                        persist_workspace()
                        log_event("scenario_applied", session_id=st.session_state.session_id,
                                  dataset_id=metadata["id"], model_version=metadata["model_version"], run_id=base["id"])
                except Exception:
                    identifier = trace_id()
                    log_event("scenario_failed", session_id=st.session_state.session_id,
                              dataset_id=metadata["id"], model_version=metadata["model_version"],
                              status="failed", error_code="WL-MODEL-503", trace=identifier)
                    st.error(f"Scenario could not be calculated; the baseline is unchanged. Reference: {identifier} (WL-MODEL-503).")
        if st.session_state.get("pending_zero_scenario"):
            with st.form("scenario_closure_confirm"):
                confirm_scenario = st.checkbox("Confirm zero customers means the kitchen is closed for this scenario.")
                confirm_scenario_submit = st.form_submit_button("Confirm zero-customer scenario")
            if confirm_scenario_submit:
                if confirm_scenario:
                    pending = st.session_state.pending_zero_scenario
                    changed_inputs = dict(base["inputs"], expected_customers=0, temperature=pending["temperature"], has_event=pending["event"], menu_items=pending["items"])
                    try:
                        result = forecast(models, df, changed_inputs)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.session_state.scenario = {"baseline_run_id": base["id"], "inputs": changed_inputs, "forecast": result, "generated_at": dt.datetime.now().astimezone().isoformat()}
                            persist_workspace()
                            log_event("scenario_applied", session_id=st.session_state.session_id,
                                      dataset_id=metadata["id"], model_version=metadata["model_version"], run_id=base["id"])
                    except Exception:
                        identifier = trace_id()
                        log_event("scenario_failed", session_id=st.session_state.session_id,
                                  dataset_id=metadata["id"], model_version=metadata["model_version"],
                                  status="failed", error_code="WL-MODEL-503", trace=identifier)
                        st.error(f"Scenario could not be calculated; the baseline is unchanged. Reference: {identifier} (WL-MODEL-503).")
                st.session_state.pending_zero_scenario = None
                st.rerun()
        scenario = st.session_state.scenario
        if scenario:
            changes = []
            for field, label in [("expected_customers", "customers"), ("temperature", "temperature"), ("has_event", "event"), ("menu_items", "menu")]:
                if scenario["inputs"][field] != base["inputs"][field]:
                    changes.append(label)
            if changes:
                st.caption("Changed inputs: " + ", ".join(changes))
            bf, sf = base["forecast"], scenario["forecast"]
            st.markdown("**Baseline** | **Scenario** | **Change**")
            demand_delta = sf["demand"] - bf["demand"]
            demand_accessible = f"{abs(demand_delta):.0f} {'more' if demand_delta > 0 else 'fewer'} meals" if demand_delta else "no change"
            st.write(f"Demand: {bf['demand']:.0f} meals | {sf['demand']:.0f} meals | {demand_delta:+.0f} meals ({demand_accessible})")
            bw, sw = sum(i["waste_kg"] for i in bf["items"]), sum(i["waste_kg"] for i in sf["items"])
            waste_delta = sw - bw
            waste_accessible = f"{abs(waste_delta):.2f} {'more' if waste_delta > 0 else 'less'} kg" if waste_delta else "no change"
            st.write(f"Waste: {bw:.2f} kg | {sw:.2f} kg | {waste_delta:+.2f} kg ({waste_accessible})")
            bp = recommendations(bf, base["buffer"], base["increment"], base.get("item_settings"), base.get("risk_thresholds"))
            sp = recommendations(sf, base["buffer"], base["increment"], base.get("item_settings"), base.get("risk_thresholds"))
            def portion_total(frame):
                values = pd.to_numeric(frame["Recommended portions"], errors="coerce").dropna()
                return int(values.sum()) if len(values) else None
            bt, stotal = portion_total(bp), portion_total(sp)
            if bt is None or stotal is None: st.write("Recommended portions: not fully available for one or more items.")
            else:
                portion_delta = stotal - bt
                accessible_delta = f"{abs(portion_delta)} {'more' if portion_delta > 0 else 'fewer'} portions" if portion_delta else "no change"
                st.write(f"Recommended portions: {bt} | {stotal} | {portion_delta:+d} portions ({accessible_delta})")
            st.caption("Scenario is an estimate under the same model; it does not change the baseline or saved plan.")
            if st.button("Promote scenario to plan inputs"):
                st.session_state.pending_promotion = True
            if st.session_state.get("pending_promotion"):
                st.warning("Promoting replaces the current baseline with this scenario.")
                pc1, pc2 = st.columns(2)
                if pc1.button("Confirm promotion", type="primary"):
                    st.session_state.baseline = {"id": uuid.uuid4().hex[:8], "generated_at": dt.datetime.now().astimezone().isoformat(), "dataset_id": metadata["id"],
                                                 "model_version": metadata["model_version"], "request_key": request_key(metadata["id"], metadata["model_version"], scenario["inputs"], base["buffer"], base["increment"], base.get("item_settings"), base.get("risk_thresholds")),
                                                 "inputs": scenario["inputs"], "forecast": scenario["forecast"], "buffer": base["buffer"], "increment": base["increment"],
                                                 "item_settings": base.get("item_settings", {}), "risk_thresholds": base.get("risk_thresholds")}
                    st.session_state.scenario = None
                    st.session_state.pending_promotion = False
                    st.session_state.saved_overrides = {}
                    st.session_state.current_plan = {}
                    st.session_state.override_reason = "No override"
                    persist_workspace()
                    log_event("scenario_promoted", session_id=st.session_state.session_id,
                              dataset_id=metadata["id"], model_version=metadata["model_version"], run_id=st.session_state.baseline["id"])
                    st.rerun()
                if pc2.button("Keep baseline"):
                    st.session_state.pending_promotion = False
                    st.rerun()
        st.button("Reset scenario", disabled=st.session_state.scenario is None,
                  on_click=reset_scenario_callback)

with tabs[2]:
    st.subheader("Historical insights")
    hist = df.copy(); hist["date"] = pd.to_datetime(hist.date)
    min_d, max_d = hist.date.min().date(), hist.date.max().date()
    default_start = max(min_d, max_d - dt.timedelta(days=INSIGHTS_DEFAULT_DAYS - 1))
    filter_suffix = metadata["id"]
    date_key, meal_key, item_key = f"hist_dates_{filter_suffix}", f"hist_meals_{filter_suffix}", f"hist_items_{filter_suffix}"
    if st.button("Clear filters", key=f"clear_hist_{filter_suffix}"):
        st.session_state[date_key] = (default_start, max_d)
        st.session_state[meal_key] = all_meals
        st.session_state[item_key] = all_items
        st.rerun()
    dates = st.date_input("Date range", value=(default_start, max_d), min_value=min_d, max_value=max_d, key=date_key)
    meal_filter = st.multiselect("Meal periods", all_meals, default=all_meals, key=meal_key)
    item_filter = st.multiselect("Items", all_items, default=all_items, key=item_key)
    date_range_ready = isinstance(dates, (tuple, list)) and len(dates) == 2
    st.caption(f"Filters active · {len(meal_filter)} meal period(s) · {len(item_filter)} item(s) · {dates[0] if date_range_ready else 'select both dates'} to {dates[1] if date_range_ready else ''}")
    if date_range_ready:
        hist = hist[(hist.date.dt.date >= dates[0]) & (hist.date.dt.date <= dates[1])]
    hist = hist[hist.meal_type.astype(str).isin(meal_filter) & hist.menu_item.astype(str).isin(item_filter)]
    if hist.empty:
        st.info("No records match these filters. Adjust the date range, meal periods or items.")
    else:
        granularity = st.radio("Trend interval", ["Daily", "Weekly"], horizontal=True)
        frequency = "D" if granularity == "Daily" else "W-SUN"
        trend = hist.groupby(pd.Grouper(key="date", freq=frequency)).agg(
            waste_kg=("food_waste_kg", "sum"), prepared=("meals_prepared", "sum"), served=("meals_served", "sum"))
        full_index = pd.date_range(trend.index.min(), trend.index.max(), freq=frequency)
        trend = trend.reindex(full_index)
        trend["preparation_gap"] = trend["prepared"] - trend["served"]
        if len(trend) < 7:
            st.caption(f"Fewer than seven {granularity.lower()} points; showing a table instead of a trend line.")
            st.table(trend.fillna(0).reset_index(names="Period"))
        else:
            figure, (waste_axis, gap_axis) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                                         gridspec_kw={"height_ratios": [3, 2]})
            waste_axis.plot(trend.index, trend["waste_kg"], color=FOREST, linewidth=2, marker="o", markersize=3)
            waste_axis.set_ylabel("Food waste (kg)")
            waste_axis.grid(axis="y", alpha=.2)
            gap_axis.bar(trend.index, trend["preparation_gap"], color=OCHRE, width=.8)
            gap_axis.axhline(0, color="#555555", linewidth=.8)
            gap_axis.set_ylabel("Prepared − served (portions)")
            gap_axis.grid(axis="y", alpha=.2)
            style_chart(figure, (waste_axis, gap_axis))
            figure.autofmt_xdate()
            figure.tight_layout()
            st.pyplot(figure, width="stretch", clear_figure=True)
            plt.close(figure)
            st.caption(f"{granularity} waste trend: {trend['waste_kg'].notna().sum()} observed points; total {trend.waste_kg.sum(min_count=1):.2f} kg. Missing periods remain gaps.")
            st.caption(f"Preparation gap: {trend.preparation_gap.sum(min_count=1):.0f} portions. Positive values indicate prepared portions left unserved.")
        rank = hist.groupby("menu_item", as_index=False).agg(total_waste_kg=("food_waste_kg", "sum"), average_waste_kg=("food_waste_kg", "mean")).sort_values(["total_waste_kg", "menu_item"], ascending=[False, True])
        st.subheader("Highest-waste items")
        rank_search = st.text_input("Search historical items", placeholder="Find an item")
        visible_rank = rank[rank.menu_item.str.contains(rank_search, case=False, na=False)] if rank_search else rank.head(10)
        for _, item_row in visible_rank.iterrows():
            with st.container(border=True):
                item_col, total_col, avg_col = st.columns([2, 1, 1])
                item_col.markdown(f"**{item_row['menu_item']}**")
                total_col.metric("Total waste", f"{item_row['total_waste_kg']:.2f} kg")
                avg_col.metric("Average", f"{item_row['average_waste_kg']:.2f} kg")
        if not rank_search and len(rank) > 10: st.caption(f"Showing top 10 of {len(rank)} items. Search to find another item.")
        st.download_button("Download historical item summary (CSV)", csv_download_bytes(rank), "wastelens_historical_items.csv", "text/csv")

with tabs[3]:
    st.subheader("Saved plans")
    saved_runs = list_runs()
    if not saved_runs:
        st.info("No saved forecast plans yet. Generate a plan to add it to local history.")
    else:
        history = pd.DataFrame([{
            "Run ID": row["run_id"], "Target date": row["target_date"], "Meal": row["meal_type"],
            "Generated": row["generated_at"], "Demand (meals)": round(row["demand"], 1),
            "Predicted waste (kg)": round(row["waste_kg"], 2), "Dataset ID": row["dataset_id"]
        } for row in saved_runs])
        st.caption("Plans and final quantities are stored in the local SQLite workspace and survive browser refreshes and app restarts. Replacing the active dataset keeps its earlier saved plans; Clear data removes all local plans and data.")
        st.dataframe(history, hide_index=True, width="stretch")
        st.download_button("Download saved plan history (CSV)", csv_download_bytes(history), "wastelens_saved_plans.csv", "text/csv")
        selected_run = st.selectbox("Inspect saved plan", [row["run_id"] for row in saved_runs])
        selected = next(row for row in saved_runs if row["run_id"] == selected_run)
        st.json(selected["payload"], expanded=False)

with tabs[4]:
    st.subheader("Data and model details")
    st.write({"dataset_id": metadata["id"], "sha256": metadata["checksum"], "model_version": metadata["model_version"], "algorithms": metadata.get("algorithms"), "library_versions": metadata.get("library_versions"), "rows": metadata["rows"], "provenance": metadata["provenance"], "date_range": [str(df.date.min()), str(df.date.max())]})
    st.write("The displayed baseline is a grouped historical median from the training portion of a chronological date holdout, using meal and weekday groups (plus item for waste). These offline metrics do not guarantee live accuracy.")
    st.write("No external weather, POS, inventory, or customer-identifying data is fetched or stored. Missing optional weather values use the training median where available; manual temperature is used for forecasts.")
    st.write("Item serving ratios are estimated from historical item portions relative to the highest-served item for that meal and date. If that history is insufficient, portions are marked unavailable rather than invented.")
