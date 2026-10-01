# WasteLens 🍲

WasteLens is a local Streamlit decision-support prototype for food-service kitchens. It validates historical CSV data, evaluates demand and waste models on a chronological date holdout, and creates an item-level preparation plan with visible data limitations.

**Synthetic demo data is illustrative only. Do not use its forecasts for operational decisions.** The app does not connect to POS, inventory, weather, supplier, or kitchen-control systems. Forecasts are not guarantees and do not replace operator judgment.

## Run locally

Use Python 3.10–3.12 with the pinned dependencies. From the project directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

For local work, run with `WASTELENS_ENV=development` (Windows PowerShell: `$env:WASTELENS_ENV="development"`). Choose **Load synthetic demo** for a clearly labeled walkthrough, or upload your own data. The regular CSV path expects the item-level schema below and enables forecasts. A separate **Import daily aggregate data (insights only)** path accepts CSV/XLSX and lets you map a date, measured waste columns in kg, and optional diner/menu fields. It displays historical aggregates only; it does not invent item-level rows or generate preparation recommendations. WasteLens saves the active dataset and planning state in SQLite at `.wastelens/wastelens.sqlite3`; **Clear data** requires confirmation and deletes the active dataset and saved forecast history. This is one shared workspace for all authorized users, not separate accounts or workspaces.

## Authentication and deployment

The Docker image sets `WASTELENS_ENV=production`; the app blocks access until Google OIDC is configured and the signed-in email is on the allowlist. Streamlit uses OIDC through `st.login()` and requires a Google OAuth client, redirect URL, cookie secret, and provider metadata. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` for local deployment configuration, or add a Render secret file named `secrets.toml` with the same TOML contents. Never commit or include real secrets in an image. Set the allowed email list to the intended operators. All allowed users share the same database and active dataset.

The included `render.yaml` is a Render Docker blueprint with a persistent 1 GB disk for SQLite. It uses Render's paid `starter` service plan. To deploy, first place this project in a private GitHub repository, create the Render service from that repository, add the Google credentials and allowlist as a secret file, then register `https://<your-service>.onrender.com/oauth2callback` as the Google OAuth redirect URI and in the secret file. Review the current Render price before creating the service; this repository does not deploy itself or create hosting resources. See [Render Docker services](https://render.com/docs/docker), [persistent disks](https://render.com/docs/disks), and [Streamlit authentication](https://docs.streamlit.io/develop/concepts/connections/authentication).

To store the database elsewhere, set `WASTELENS_DB_PATH` to a writable file path before starting the app. For Docker, mount a persistent volume at `/app/.wastelens` to retain data between container recreations. The app is a single-workspace installation: users who can access the same instance share its active dataset and saved plans.

To create/update the optional local model artifacts and evaluation metadata from the demo dataset:

```bash
python train_models.py
```

The app trains from the currently loaded dataset in memory, so these saved artifacts are not required to run it.

## Data format

Required columns:

| Column | Meaning |
|---|---|
| `date` | Historical service date (`YYYY-MM-DD`) |
| `meal_type` | Meal period, such as Lunch or Dinner |
| `menu_item` | Item name |
| `meals_prepared` | Portions prepared; must be at least portions served |
| `meals_served` | Portions served for this item |
| `food_waste_kg` | Waste recorded for this item in kilograms |

Optional columns: `temperature` (°C), `holiday` (0/1), `event` (0/1), `expected_customers` (non-negative count), and `portion_mass_kg` (positive mass of one portion of that item). Approved aliases are shown before import; ambiguous mappings block the import. Decimal-comma CSVs are supported. Waste in grams is converted only when an explicit gram unit column is supplied; unknown units reject the affected row. Missing optional weather/condition values are imputed to a neutral/default value; missing customer counts are estimated from the maximum item servings for the date and meal. The app labels these assumptions.

Uploads are limited to 10 MB, UTF-8 CSV, at least 14 distinct usable dates, and at least 80% valid rows. Invalid rows and duplicate date/meal/item keys are rejected and can be downloaded for review. Duplicate keys block by default; an explicit keep-last confirmation is required to resolve them. Fourteen to 29 days of usable history produce a degraded-quality warning.

## How forecasts work

- Demand is forecast per meal period. Historical demand target is the maximum item-level `meals_served` for each date and meal, because the provided schema has item-level servings but no separate meal headcount.
- Item waste is forecast from historical item rows.
- A chronological date holdout compares Random Forest and XGBoost and reports MAE, RMSE, R², WAPE, stability across temporal blocks, and a grouped historical-median baseline (meal and weekday; waste also groups by item). Lag and rolling features use only observations before the forecast date. The app flags results when the selected model does not beat that baseline.
- Item portions use the median historical ratio of an item's servings to the highest-served item for that meal and date, matching the demand target used in training. If that ratio cannot be estimated, the app displays “Not available” rather than fabricating a quantity.
- Waste risk percentages use item-level historical `portion_mass_kg` to convert planned portions into kilograms. If this optional measurement is absent, risk is “Not available” rather than a dimensionally invalid comparison of kilograms and portions.
- Demand and item-waste drivers use per-run SHAP where supported. If the explanation engine fails, the app explicitly labels global feature importance as a fallback.
- Recommendations apply a visible safety buffer and production increment. Users can enter current planned portions, inspect a bounded potential portion reduction, record an override reason, reset quantities, and export the final plan. Portion reduction is not reported as kilograms of waste avoided.
- Per-item production increments and capacity limits can be configured. Capacity-limited recommendations show the unmet-demand risk and uncapped quantity. Identical forecast requests use an atomic idempotency key and recover the original saved run.
- Out-of-training-range customer counts or temperatures produce a lower-confidence warning. Scenarios do not overwrite the baseline.

Metrics on small or synthetic datasets are not evidence of real-world performance. The current MVP does not provide calibrated confidence intervals, per-user workspace separation, automated retraining, or live POS, inventory, supplier, weather, or kitchen-control integrations. Those connections require a selected provider, credentials, and an agreed data contract. Daily aggregate imports cannot be used to train the item-level planning models. The [Finnish school food-waste dataset](https://zenodo.org/records/20825233) is one example of a real daily aggregate workbook that fits the insights-only path; inspect the data license and definitions before relying on it.

## Included fixtures

`data/test_datasets/` includes sample valid, sparse, invalid-row, and adversarial CSVs for manual QA. `data/synthetic_historical_data.csv` is generated by `generate_data.py` and is synthetic. `data/salary_data.csv` is unrelated to WasteLens and is not used by the app.

## Verification status

Automated acceptance checks live in `tests/test_wastelens_acceptance.py` and can be run with:

```bash
python -m unittest discover -s tests -v
```

The checks cover the golden, missing-column, invalid-row, alias, duplicate, unit-conversion, sparse-history, temporal-model, prediction, recommendation, explanation-fallback, idempotency, persistence, and clear-data paths. The test suite does not replace the PRD's manual keyboard/accessibility check, 360/768/1440 px browser matrix, 200% zoom check, five-user usability study, or operational pilot using real post-service waste measurements. Do not claim those outcomes until they are performed and recorded. No real-world savings or model accuracy is asserted from synthetic fixtures.
