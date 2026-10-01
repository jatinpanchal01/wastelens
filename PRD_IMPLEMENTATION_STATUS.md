# PRD implementation and release status

This file separates implemented software behavior from evidence that requires a live UI, representative operators, or real kitchen data. Synthetic fixtures are not operational proof.

## Automated implementation coverage

| PRD area | Implemented behavior | Evidence |
| --- | --- | --- |
| F1 Data intake | 10 MB CSV limit, required-field checks, approved alias preview, decimal-comma parsing, explicit gram-to-kg conversion, future-date rejection, row-level rejection export and valid/warning/rejected counts, duplicate confirmation, 14-day minimum, degraded-history warning, template download and import progress state | `tests/test_wastelens_acceptance.py` validation checks; app import flow |
| F2 Demand and waste | Date-aware lag and rolling features, chronological holdout, Random Forest/XGBoost comparison, baseline comparison, nonnegative and customer-plausibility checks, forecast version and input metadata, saved result recovery | Model and forecast acceptance checks |
| F3 Preparation plan | Buffer and increment rounding, configurable item increments and capacities, unmet-demand risk, item waste risk with valid portion-mass units, explanation fallback, editable final quantities and recorded reason | Recommendation and SHAP-fallback acceptance checks |
| F4 What-if | Separate scenario result tied to baseline run, changed-input and delta display, zero-customer confirmation, explicit promotion confirmation, retry error state preserves baseline | App implementation; visual interaction check remains open |
| F5 Trends/reports | Daily/weekly view, 90-day default, date filters/reset, empty-period gaps, unit-labeled Matplotlib charts, CSV formula escaping, PDF generation with font selection and CSV fallback | App implementation and AppTest rendering smoke; PDF download/failure check remains open |
| Persistence/idempotency | SQLite workspace and run history, active-dataset replacement detection, atomic unique request-key behavior, restore after refresh, confirmed clear action | Persistence lifecycle and duplicate-request checks |
| Observability/privacy | Structured events include status, versions, duration and trace IDs; uploaded rows are not logged | `wastelens_observability.py` and instrumented app events; deployment log review remains open |
| Authentication/deployment prep | Google OIDC login, verified-email allowlist gate, logout, production fail-closed behavior, Render Docker blueprint, persistent disk and secret-file exclusions | Docker smoke and config review; live OAuth/deployment still requires owner credentials, repository, and hosting account |
| Alternate aggregate schema | CSV/XLSX mapping for daily waste totals, diner count and menu identifier; trend-only view with no item forecasting | Parser mapping smoke; confirm with the released Finnish workbook before operational interpretation |
| Documentation/setup | Local install and Docker volume instructions, CSV schema, assumptions, synthetic-data warning, limitations, acceptance command | `README.md` |
| Optional model artifacts | Demo-only model and feature artifacts with dataset checksum, model version, candidate comparison metrics, and Python/numpy/pandas/scikit-learn/XGBoost versions | `train_models.py`; refreshed `models/` artifacts use the labeled synthetic fixture |

## Verification run

- Syntax compilation: passed for the app, core, store, configuration, observability, and acceptance suite.
- Automated acceptance suite: **16 passed**. Run with `python -m unittest discover -s tests -v` after installing `requirements.txt`.
- Acceptance suite on the rebuilt production image with Streamlit 1.64.0: **16 passed** (test fixtures mounted read-only; test process explicitly ran in development mode).
- Aggregate schema smoke: Finnish-style `Date`, `PW`, `KSW`, `DINERS`, and `MENU` headings mapped correctly; 2.7 kg total and per-diner calculations matched a hand-checked fixture. Aggregate-mode AppTest restored a persisted aggregate workspace, displayed the insights-only warning, and had no script exceptions.
- Production authentication gate smoke: with no OIDC secrets, Streamlit displayed the setup error and stopped without a script exception.
- Production Docker image: build succeeded; local container health endpoint returned `ok` and the app served its page. Container was stopped after the smoke check.
- Deployment packaging audit: build context excluded local model artifacts (~5.2 MB), QA fixtures/tests, docs, training scripts, local SQLite state, secrets, and unrelated salary data; Dockerfile copies only app runtime modules, Streamlit config, pinned dependencies, and the labeled synthetic demo CSV. The clean image's file inventory confirmed these exclusions, and health returned `ok`.
- Docker release smoke: `docker build -t wastelens:prd-check .` succeeded; a temporary container returned `ok` from `/_stcore/health` and served the app page. The smoke-test container was stopped and removed.
- After the 1 October UI refresh in `app.py` and `wastelens_theme.py` (including a 44 px minimum button height), the acceptance suite again passed (**16 tests**) and the rebuilt Docker image again passed health and app-page smoke checks.
- Streamlit AppTest empty-state smoke: passed with no script exceptions; WasteLens title and upload control were present.
- Streamlit AppTest end-to-end smoke: labeled synthetic demo -> dashboard -> generated/persisted plan passed; an injected inference failure preserved the successful plan and displayed `WL-MODEL-503`; scenario reset preserved the exact baseline.
- Blank-screen investigation: the supplied browser log showed a Vega-Lite 5.20.1 spec mismatch against Streamlit's bundled 5.9.3 renderer and charts with non-finite extents. Replaced the Vega-backed Streamlit chart widgets with Matplotlib charts; reran all 16 tests successfully.
- Synthetic fixture model training: 900 rows trained and evaluated in 1.88 seconds in this environment. This is a local measurement, not the PRD's cold-start or prediction p95 benchmark.
- Runtime used in this workspace: bundled CPython 3.12 with this project's installed site-packages. The checked-in `.venv` launcher points at a Python 3.12 installation path that is absent on this machine; do not use that stale launcher here.
- Visual browser verification could not be completed because the computer-use browser security policy blocked navigation to the local preview. AppTest verifies script-level rendering and state but does not replace visual, browser-matrix, or assistive-technology checks.

## Release gates still requiring evidence

| Gate | State | What closes it |
| --- | --- | --- |
| P0-01 Golden CSV | Validator test passes; end-to-end demo path passes | Run upload confirmation with the golden CSV in browser QA |
| P0-02 Missing column | Automated validator test passes | Confirm the exact missing-field message in browser QA |
| P0-03 Generate twice | Atomic storage idempotency regression passes | Repeat an identical plan in browser QA and confirm the run ID is reused |
| P0-04 Prediction failure | AppTest fault injection passes; prior successful run remains with `WL-MODEL-503` | Confirm the user-facing failure state in browser QA |
| P0-05 Recommendation math | Formula, rounding, capacity, and risk unit tests pass | Review quantities on approved operator data |
| P0-06 Scenario reset | AppTest regression passes | Keep the baseline values, run ID, and saved plan unchanged after resetting a scenario |
| P1-01 Explanation fallback | Automated forced-SHAP-failure test passes | Confirm the fallback text is visible in the running UI |
| P1-02 Responsive layout | CSS and responsive Streamlit layout implemented; not visually checked | Test latest Chrome, Edge, and Firefox at 360, 768, and 1440 px, 320 px, and 200% zoom; confirm no page overflow or hidden critical action |
| P1-03 PDF failure fallback | CSV remains available before PDF generation and exceptions show a fallback message; not fault-injected through the UI | Force PDF failure and verify CSV and on-screen plan remain usable |
| Accessibility | Labels, units, textual risk and minimum button sizing implemented; not manually audited | Keyboard sequence, focus visibility, screen reader labels, chart summaries, and contrast check |
| Live identity/deployment | Not deployed; auth gate and blueprint are prepared | Create/authorize private source repository, configure Google OAuth client and allowed emails, review paid Render plan, create service and verify full redirect/login/logout flow |
| Performance/compatibility | Not measured | Record cold start, prediction p95, interaction response, and browser matrix on demo hardware |
| Five-user usability and trust | Not run | Observe five representative operators and record task completion/ease/trust results |
| Operational waste impact | Not measured | Pilot with real post-service waste and comparable baseline period; distinguish measured from estimated savings |
| Presentation/demo rehearsal | Not performed | Capture screenshots from released build and rehearse the end-to-end demo |

Live POS, inventory, suppliers, weather services, identity/access control, and automated retraining are explicitly excluded from the PRD MVP. Connecting them would require approved providers, credentials, contracts, and a scope change; this implementation does not claim those integrations exist.
