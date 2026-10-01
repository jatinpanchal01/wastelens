# WasteLens product requirements document

**Product:** WasteLens — AI-powered food waste prediction and smart kitchen advisor  
**SDG:** 12 — Responsible Consumption and Production  
**Release:** Seven-day MVP / demo build  
**Version:** 1.0  
**Date:** 29 September 2026  
**Target submission:** 9 October 2026

> **Product thesis:** Predict -> explain -> recommend. WasteLens is valuable only when a forecast becomes a practical preparation decision.

---

## Contents

1. [How to use this document](#how-to-use-this-document)
2. [Executive summary](#executive-summary)
3. [Purpose, goals and urgency](#purpose-goals-and-urgency)
4. [Scope and release boundaries](#scope-and-release-boundaries)
5. [Target users](#target-users)
6. [User benefits and experience principles](#user-benefits-and-experience-principles)
7. [Core journeys and information architecture](#core-journeys-and-information-architecture)
8. [Feature requirements](#feature-requirements)
9. [AI/ML requirements](#aiml-requirements)
10. [Data and service contracts](#data-and-service-contracts)
11. [Responsive UI specification](#responsive-ui-specification)
12. [Non-functional requirements](#non-functional-requirements)
13. [Cross-feature edge cases](#cross-feature-edge-cases)
14. [QA strategy](#qa-strategy)
15. [Success metrics](#success-metrics)
16. [Seven-day delivery plan](#seven-day-delivery-plan)
17. [Risks, open decisions and glossary](#risks-open-decisions-and-glossary)

---

## How to use this document

WasteLens helps a food-service kitchen decide how much to prepare tomorrow, what is likely to be wasted, and which operational action to take.

**Product thesis**
Predict -> explain -> recommend. The product is valuable only when a forecast becomes a practical preparation decision.

### Document status

| Field | Value |
| --- | --- |
| **Owner** | Product team |
| **Release** | MVP / demo build |
| **Target date** | 9 October 2026 |
| **Primary surface** | Responsive Streamlit web app |
| **Primary stack** | Python, Pandas, scikit-learn or XGBoost, SHAP, Streamlit |

### Decision hierarchy

1. **Must:** upload valid historical data; view demand and waste predictions; receive item-level preparation recommendations.
2. **Should:** test a what-if scenario; understand main prediction drivers; inspect trends.
3. **Could:** download a report and use optional weather inputs.
4. **Won't in MVP:** procurement automation, donation logistics, IoT scales, live POS integration, multi-location tenancy.

### Assumptions

- One kitchen/location per app instance.
- Historical data is supplied as CSV; synthetic enrichment is clearly labeled.
- Predictions support decisions and do not autonomously change production.
- Kilograms and portions are the only MVP quantity units.

---

## Executive summary

Busy kitchen professionals lose money and food when preparation plans are based on habit, incomplete attendance signals, or yesterday's intuition. WasteLens converts historical operating data and today's conditions into an explainable next-day plan.

**Flow:** Historical data -> Forecast -> Waste risk -> Action plan

**Problem:**

Overproduction creates avoidable waste; underproduction risks stock-outs and service failure.

**Response:**

Predict meals consumed and item-level waste, then recommend preparation quantities with reasons.

**Proof:**

Show model metrics, back-tested results, user feedback, and estimated waste avoided.

### MVP outcome

A kitchen operator can import a validated dataset, choose tomorrow's conditions, generate a forecast, review an item-by-item plan, adjust one scenario, and download a decision summary in under five minutes.

### Product constraints

| Constraint | Product response |
| --- | --- |
| Seven calendar days | Single location, CSV-first ingestion, one prediction workflow, no authentication. |
| Limited real data | Display dataset provenance, data-quality warnings and honest low-confidence states. |
| Explainability expected | Show global feature importance and per-prediction drivers; retain rule-based safeguards. |
| Responsive, attractive UI | Design mobile-first cards, accessible charts, persistent actions and explicit states. |

---

## Purpose, goals and urgency

### Context

Food-service teams routinely balance uncertain demand against the operational cost of running out. Busy professionals often work from spreadsheets, memory, or fixed buffer percentages. These methods hide item-level risk and offer little evidence for why a plan should change.

### Problem statement

When demand signals, menu mix, calendar effects and prior waste are not evaluated together, operators over-prepare low-demand items, react too late to waste patterns, and cannot explain decisions to managers.

### Urgency

The project has a seven-day build window and a 9 October 2026 submission. Scope must favor a stable end-to-end demo and credible quality evidence over broad integrations.

**North-star question:**

Given historical operations and tomorrow's conditions, how much should we prepare, what waste should we expect, and what should we change?

### Goals

1. Generate next-day meal-demand and waste forecasts.
2. Translate forecasts into safe, item-level preparation quantities.
3. Expose the main reasons, confidence and limitations behind the result.
4. Let users compare a changed scenario without overwriting the base prediction.
5. Present all critical actions clearly on phone, tablet and desktop.

### Non-goals

**No autonomous control**

WasteLens does not place orders or change kitchen production.

**No redistribution network**

Donation matching, pickup and compliance are outside MVP.

**No causal guarantee**

Feature importance explains model influence, not causation.

---

## Scope and release boundaries

| Priority | Included | Exit condition |
| --- | --- | --- |
| **Must** | CSV upload and validation; dashboard; demand forecast; item waste forecast; preparation recommendation | Happy path works with approved demo dataset and one degraded-data case. |
| **Should** | What-if scenario; model explanation; historical trend and high-waste ranking | Results update without damaging base forecast; explanations have plain language. |
| **Could** | PDF/CSV report; optional weather field; model comparison panel | Only after Must flows pass regression testing. |

### In scope

- One kitchen and timezone per run.
- Daily and item-level historical records.
- Manual future inputs: date, expected customers, menu, holiday/event, temperature.
- Regression-based demand and waste estimates.
- Risk thresholds, preparation recommendations and scenario comparison.
- Local file persistence or session state suitable for a demo.

### Explicitly out of scope

- User accounts, roles, payments or subscriptions.
- Live point-of-sale, inventory, supplier or delivery APIs.
- Automated retraining and model registry.
- Real-time sensor ingestion and computer vision.
- Meal donation, compliance or transport logistics.
- Native mobile apps and offline-first synchronization.

**Scope change rule:** a new feature enters the seven-day MVP only if an existing feature of equal effort is removed and Must-path QA remains scheduled for Day 7.

---

## Target users

### Priya - operations manager

**Context:** Manages a corporate cafeteria or cloud-kitchen shift while coordinating demand, staffing and cost.

**Job:** Approve tomorrow's preparation plan in less than five minutes.

**Pain:** Data is scattered; fixed buffers create waste; management asks for evidence.

**Need:** One summary with forecast, risk, action and confidence.

### Arjun - head chef

**Context:** Converts expected attendance into batch quantities across rice, dal, roti and vegetables.

**Job:** Prepare enough without sacrificing service quality.

**Pain:** Meal forecasts do not translate into ingredient-level action.

**Need:** Item-level quantities, safe rounding and override visibility.

### Maya - sustainability analyst

**Context:** Reviews weekly performance and reports progress against SDG 12 goals.

**Job:** Identify recurring waste drivers and quantify improvement.

**Pain:** Waste logs lack trends, model context and decision history.

**Need:** Comparable metrics, exportable summaries and provenance.

### Ravi - remote site supervisor

**Context:** Checks the plan on a phone between service tasks and has uneven connectivity.

**Job:** Confirm the latest plan and spot high-risk items quickly.

**Pain:** Dense dashboards and tiny charts fail on mobile.

**Need:** Stacked cards, concise labels and recoverable network states.

### Permission model for MVP

All personas use one local workspace without authentication. The interface labels calculated values, manual inputs and user overrides distinctly. Multi-user access control is deferred; the demo must not imply that access is secured.

---

## User benefits and experience principles

| Pain point | WasteLens response | Expected benefit |
| --- | --- | --- |
| Guesswork and fixed buffers | Demand prediction plus confidence range | More consistent planning |
| Waste reported too late | Item risk ranking before production | Earlier intervention |
| Forecasts lack action | Recommended quantities and change percentage | Faster decision-making |
| Black-box skepticism | Drivers, model metrics and data warnings | Higher trust and safer use |
| Spreadsheet friction | Validated import, reusable session and report | Less manual analysis |
| Desktop-only tools | Responsive, touch-friendly layout | Usable during kitchen operations |

### Experience principles

**Action before analysis**

Lead with tomorrow's plan; put diagnostic charts one level deeper.

**Honest uncertainty**

Never present a low-quality forecast as a precise fact.

**Human in control**

Recommendations are editable; overrides are visible and never retrain the model silently.

**Recover without losing work**

Validation and retry states preserve usable inputs and base predictions.

**Color is not the message**

Risk uses labels, icons or patterns in addition to color.

**One decision vocabulary**

Prepared, served, wasted, recommended and overridden mean the same everywhere.

---

## Core journeys and information architecture

1. **Load data:** User uploads a CSV or opens the approved demo dataset.
2. **Resolve quality:** System validates schema, types, duplicates and usable history; blocking defects are actionable.
3. **Set tomorrow:** User chooses date, expected customers, menu and optional conditions.
4. **Generate:** System calculates demand, item waste, risk and recommended preparation in one versioned run.
5. **Review and simulate:** User inspects drivers, changes a scenario input and compares it with the unchanged baseline.
6. **Act or share:** User accepts/overrides quantities and downloads a summary.

**Flow:** Data -> Plan inputs -> Forecast -> Compare -> Export

### Information architecture

| Field | Value |
| --- | --- |
| **Overview** | KPIs, risk, recommendations |
| **Plan tomorrow** | Inputs and prediction run |
| **What-if** | Baseline/scenario comparison |
| **Insights** | Trends and feature drivers |
| **Data & model** | Import, quality, metrics |

### Critical state model

**No data** -> Upload prompt

**Invalid data** -> Blocking errors + row report

**Ready** -> Planning form enabled

**Running** -> Controls locked; cancel hidden in MVP

**Success** -> Versioned result displayed

**Degraded** -> Result with warning and limitations

**Failed** -> Retry without clearing valid inputs

**Demo golden path:** sample dataset -> default future inputs -> Generate plan -> change expected customers -> Compare -> Download summary.

---

## Feature requirements

### F1 — Data intake and validation

**What it does:** accepts a historical CSV, validates its structure and creates an analysis-ready dataset. Required fields: `date, meal_type, menu_item, meals_prepared, meals_served, food_waste_kg`. Optional fields: temperature, holiday, event, expected_customers, and `portion_mass_kg` (positive kilogram mass for one portion of that item; required to calculate a dimensionally valid waste-rate risk).

#### Flow and integration

1. User drags or selects one **.csv** file, maximum 10 MB.
2. Backend reads UTF-8, detects delimiter, maps exact or approved aliases, then validates each row.
3. UI displays counts for valid, warning and rejected rows plus a 20-row preview.
4. User confirms import. Clean data enters the feature-engineering pipeline and enables planning.

**Interactions:** prediction, trends, explanations and report all read the same immutable dataset version and checksum.

#### UI behavior

- Drop zone includes template download and required-column list.
- Progress has Uploading, Validating and Ready labels.
- Errors appear in a row-level table downloadable as CSV.
- A replacement upload requires confirmation because it invalidates existing results.

#### Validation and fallback

| Condition | Handling / message |
| --- | --- |
| Wrong type / over 10 MB | Block: “Upload one CSV file up to 10 MB.” |
| Missing required column | Block and name it: “Missing required column: meals_served.” |
| Bad date or negative value | Reject row; allow import only if at least 80% of rows remain valid. |
| Prepared < served | Warn and reject row unless documented carry-over is supported later. |
| Duplicate key | Keep last only after explicit confirmation; default is block. |
| Too little history | Under 14 usable days: block model run. 14-29 days: degraded warning. |
| Parse failure | Preserve file choice; offer retry and template. |

**Acceptance:** A valid template imports once, displays deterministic counts, produces a dataset ID, and enables Generate. Invalid imports never trigger training or prediction.

---

### F2 — Demand and waste prediction

**What it does:** predicts meals consumed and waste kilograms for a selected date and menu. Every result is tied to dataset, model and input versions.

#### Flow and integrations

1. User enters date, expected customers, menu selection, holiday/event status and optional temperature.
2. Client validates ranges and sends one prediction request with an idempotency key.
3. Feature pipeline derives day-of-week, lag values and recent waste rates.
4. Demand model predicts consumed meals; waste model predicts each item's waste.
5. Post-processing clips impossible negatives and returns confidence/quality labels.
6. Recommendation engine consumes the same response.

#### UI behavior

- Generate button stays disabled until data and required inputs are valid.
- Loading skeleton replaces results; button cannot double-submit.
- Cards show expected customers, predicted demand, expected waste and data-quality status.
- Values show units and “generated at” time.

#### Validation, edge cases and fallback

| Condition | Required behavior |
| --- | --- |
| Past planning date | Allow as back-test; label “Historical test.” |
| Date over 30 days away | Warn: lower reliability; allow run. |
| Customers blank / <0 | Block: “Enter expected customers from 0 to 100,000.” |
| Unseen menu item | Use category fallback if trained; otherwise exclude and warn. |
| Missing optional weather | Use training median; disclose imputation. |
| Model unavailable | No invented result. Show “Prediction is unavailable. Retry or use the last successful run.” |
| Timeout after 15 s | Fail once; preserve inputs; offer Retry. |
| Duplicate submission | Return the first result for the same key. |

**Acceptance:** Same model, dataset and inputs produce the same displayed result. Predicted demand never exceeds the configured plausibility ceiling without a warning.

---

### F3 — Preparation recommendations

**What it does:** proposes the number of meals and item quantities to prepare, with change versus the user's current plan and estimated waste avoided.

#### Decision logic

```text
recommended_meals = ceil(predicted_demand * (1 + safety_buffer))
recommended_item_qty = recommended_meals * historical_qty_per_serving
estimated_avoided = max(0, planned_qty - recommended_qty)

Default safety_buffer: 5%
Rounding: item-specific production increment
```

The formula is a proposed MVP rule. Configuration and units must be displayed; model outputs are not silently altered.

#### Flow and interaction

1. Prediction response enters rules layer.
2. Rules apply safety buffer, serving ratio and production increment.
3. UI ranks items by expected waste and highlights recommended change.
4. User accepts or overrides a quantity and optionally selects a reason.
5. Export records both recommended and final values.

#### UI, errors and safeguards

| Condition | Behavior |
| --- | --- |
| No current plan | Show recommendation; omit percentage change. |
| Zero predicted demand | Recommend zero only with “Confirm closure/no service” prompt. |
| Negative model output | Clip to zero; log post-processing warning. |
| Recommendation over capacity | Cap display at capacity and show unmet-demand risk. |
| Unknown serving ratio | Show meals only; item quantity becomes “Not available.” |
| Override below forecast | Warn about shortage; allow confirmation. |
| Override non-numeric | Inline error: “Enter a quantity of 0 or more.” |
| Calculation failure | Keep forecast visible; hide action claim and explain retry. |

#### Acceptance criteria

- Rounding always uses the item's configured increment and preserves unit.
- Estimated waste avoided cannot be negative and is labeled as an estimate.
- Overrides never modify the baseline forecast or train a model.
- The user can reset all overrides to generated values in one action.

---

### F4 — Item risk and explainability

**What it does:** ranks menu items by expected waste and explains the strongest factors influencing the run. Risk combines predicted waste rate with configurable thresholds.

#### Risk classification

| Level | Default rule | UI |
| --- | --- | --- |
| Low | <5% predicted waste rate | “Low” + neutral/green accent |
| Moderate | 5-10% | “Moderate” + amber accent |
| High | >10% | “High” + red accent |

Waste rate = predicted waste quantity in kg / planned preparation quantity in kg. Convert portions with the item's `portion_mass_kg`. If portion mass is missing or the denominator is zero, level is “Not available,” never Low.

#### Flow and integration

1. Prediction returns values and feature vector.
2. Risk service applies versioned thresholds.
3. Explainability module calculates SHAP values or approved model-native importance.
4. UI shows up to three increasing and decreasing drivers with units for each demand and item-waste prediction.

#### UI and fallback

- Item list defaults to highest predicted waste; sorting is explicit.
- Chart text states “influences prediction,” not “causes waste.”
- Tooltips explain each input in plain language.
- Screen-reader label includes item, value, unit and risk.

| Failure | Fallback |
| --- | --- |
| SHAP unavailable | Show global feature importance and “Local explanation unavailable.” |
| Unsupported model | Hide driver chart; retain model metrics. |
| All item values equal | Stable alphabetical tie-break. |
| Extreme outlier | Flag outside training range; do not suppress. |
| Threshold missing | Use defaults and mark “Default policy.” |

**Acceptance:** A high-risk item is identifiable without color. Driver values reconcile with the prediction run and explanations never claim causality.

---

### F5 — What-if simulation

**What it does:** recalculates results after the user changes future conditions, then shows the difference from the saved baseline.

#### Flow

1. User opens What-if from a successful baseline run.
2. Controls copy baseline values; user changes customers, temperature, event or menu.
3. Apply scenario sends a new request linked to baseline_run_id.
4. System displays Baseline, Scenario and Delta for demand, preparation and waste.
5. Reset restores baseline inputs; Promote copies scenario to the working plan after confirmation.

#### UI behavior

- Desktop uses a two-column comparison; mobile stacks Baseline before Scenario.
- Changed inputs carry a visible “Changed” label.
- Delta uses signed values and accessible wording: “35 more meals.”
- Charts share axis scales to prevent misleading comparison.

#### Edge cases and errors

| Condition | Required behavior |
| --- | --- |
| No baseline | Disable What-if: “Generate a plan first.” |
| No fields changed | Disable Apply scenario. |
| Customers = 0 | Allow only with closure confirmation. |
| Input out of training range | Allow within hard bounds; show low-confidence warning. |
| Menu item removed | Remove from scenario only; baseline remains visible. |
| Scenario request fails | Keep baseline and edits; retry scenario only. |
| Rapid slider changes | No automatic server calls; calculate on Apply. |
| Promote after dataset replaced | Block stale action and require regeneration. |

**Acceptance:** Scenario generation never mutates the baseline. Reset is exact. All deltas use consistent units and direction.

---

### F6 — Historical insights and downloadable reports

#### Historical insights

**What it does:** shows daily/weekly waste trend, prepared-versus-served gap, and highest-waste items for the imported dataset.

**Flow:** dataset version -> aggregations -> cached chart payload -> filter by date range, meal type or item.

- Default range is all valid data up to 90 days.
- Charts include units, accessible text summary and empty states.
- Filters apply together and show active state.

**Fallback:** fewer than seven points uses a table instead of a trend line. Empty filter result says “No records match these filters” and offers Clear filters.

#### Downloadable decision report

**What it does:** exports a one-run PDF summary and machine-readable CSV of item recommendations.

**Report content:** run time; input conditions; dataset/model version; prediction and confidence; item plan; overrides; assumptions; warnings.

- Download is generated from server-side run data, not the visible DOM.
- Filename: wastelens_YYYY-MM-DD_run-shortid.pdf
- Controls are disabled while generating; second click does not duplicate.

**Fallback:** if PDF generation fails, offer CSV and plain on-screen print view. Never label a failed or partial file as complete.

#### Shared edge conditions

| Condition | Handling |
| --- | --- |
| Missing date interval | Render gaps; do not interpolate silently. |
| Mixed units | Block import or require normalized unit mapping. |
| Large number of items | Show top 10 plus search; export all. |
| Special characters | UTF-8 output; escape CSV formula prefixes and PDF text. |
| Popup blocked | Use direct download response; keep retry action. |

---

## AI/ML requirements

### Proposed pipeline

1. Normalize dates, item names and units.
2. Create calendar, lag, rolling-average, menu, temperature and holiday features.
3. Split chronologically; never randomly leak future rows into training.
4. Compare naive seasonal baseline with Random Forest and XGBoost regressors.
5. Select by validation mean absolute error (MAE), stability and inference speed.
6. Persist preprocessing and model together with version metadata.

### Outputs

- **Demand model:** predicted meals consumed.
- **Waste model:** predicted waste kilograms per menu item.
- **Explanation:** top feature contributions.
- **Quality:** model metrics and prediction confidence label.

### Required safeguards

| Requirement | Implementation expectation |
| --- | --- |
| No leakage | Fit transformers on train split only. |
| Reproducibility | Fixed random seed; record dataset checksum and library versions. |
| Non-negative output | Clip after prediction; retain raw value in logs. |
| Baseline gate | Do not claim AI value if model does not beat naive MAE. |
| Out-of-range input | Flag extrapolation and lower confidence. |
| Explainability | SHAP for selected tree model; cached per run. |

### Evaluation and release gates

**Offline metrics:**

MAE, root mean squared error (RMSE), R-squared, and weighted absolute percentage error where denominator is valid.

**Slices:**

Report by meal type, day of week and top-volume items; surface weak slices.

**Minimum gate:**

Model beats naive MAE on the same holdout and produces no schema or unit violations.

**Evidence rule:** target accuracy values are not invented in advance. Record the achieved holdout metrics in the app and presentation after training.

---

## Data and service contracts

### Core entities

| Entity | Minimum fields |
| --- | --- |
| Dataset | id, checksum, rows, date_range, quality |
| Model | version, algorithm, trained_at, metrics |
| PredictionRun | id, dataset_id, model_version, inputs, status |
| ItemPrediction | item, demand, waste_kg, risk, drivers |
| Recommendation | item, planned, recommended, unit, buffer |
| Override | recommended, final, reason, created_at |
| Scenario | baseline_run_id, changed_inputs, deltas |

### Logical service boundaries

- **Ingestion:** parse, normalize, validate, version.
- **Features:** deterministic train/inference transforms.
- **Prediction:** model loading, inference, post-processing.
- **Recommendation:** rule-based quantities and thresholds.
- **Reporting:** charts, PDF and CSV output.

For the seven-day Streamlit build these may be Python modules in one repository, not network microservices.

### Persistence

Use local versioned files or SQLite. Store no secrets in source. Uploaded data, the resumable active plan, overrides, scenarios, and forecast run history remain local and survive refresh/restart. A Clear data action deletes the local active dataset and saved run records after confirmation. The MVP is one shared local workspace without authentication; deployment must restrict access to trusted operators and document the local database path.

### Prediction request and response

```text
POST /predict
{ dataset_id, target_date, expected_customers, menu_items[],
temperature_c?, is_holiday, event_type?, idempotency_key }

200 { run_id, generated_at, model_version, data_quality,
predicted_demand, demand_range?, predicted_waste_kg,
items:[{name, waste_kg, risk, recommended_qty, unit, drivers[]}],
warnings[] }
422 { code:"VALIDATION_ERROR", field_errors:{...}, trace_id }
503 { code:"MODEL_UNAVAILABLE", retryable:true, trace_id }
```

**API rules:** numbers are JSON numbers; timestamps are ISO 8601; missing is null, never zero; error codes are stable; internal stack traces never reach the UI.

---

## Responsive UI specification

### Visual system

- **Palette:** deep green primary, off-white surfaces, charcoal text; amber and red reserved for risk.
- **Typography:** one sans-serif family; 16 px minimum body on screens; tabular numerals for KPIs.
- **Spacing:** 8 px base grid; 16 px card padding; 24-32 px section gaps.
- **Shape:** 10-12 px corners; restrained shadows; visible 1 px borders.
- **Motion:** 150-250 ms state transitions; respect reduced-motion preference.

### Breakpoints

| Viewport | Layout |
| --- | --- |
| <600 px | Single column; bottom/compact nav; full-width buttons |
| 600-1023 px | Two KPI columns; collapsible sidebar |
| >=1024 px | Four KPI columns; persistent sidebar; split comparison |

### Component behavior

- Cards never require horizontal scrolling.
- Tables become labeled item cards below 600 px; CSV remains full fidelity.
- Charts resize to container and retain axis labels; a text summary follows every chart.
- Primary action remains visible after form completion, not permanently sticky over content.
- Skeletons match final card geometry to prevent layout shift.
- Toasts announce completion; blocking errors stay inline until resolved.

### Accessibility

- WCAG 2.1 AA contrast target; 44 x 44 px touch targets.
- Visible keyboard focus and logical tab order.
- All inputs have persistent labels, units and programmatic error association.
- Risk never relies on color alone.
- Screen readers receive chart summaries and loading status.

**Streamlit note:** use native responsive columns cautiously and test custom CSS after each Streamlit upgrade. Core actions must remain usable if decorative CSS fails.

### Required UI states per screen

Initial/empty, loading, success, partial/degraded, validation error, system error, no results and stale result. QA must capture each at mobile and desktop widths.

---

## Non-functional requirements

| Area | Requirement | Verification |
| --- | --- | --- |
| Performance | Initial usable view <3 s on demo hardware; prediction p95 <5 s for approved dataset; interactions acknowledge within 200 ms. | Timed cold/warm runs; throttled UI check. |
| Reliability | No partial result is labeled success. Retry is idempotent. Last successful run remains readable after a new failure. | Fault injection and repeat requests. |
| Data integrity | Dataset checksum, model version and units persist with every run and export. | Contract and export assertions. |
| Privacy | No personal customer data is required. Uploaded files remain local; clear-data action removes them. | File-system and UI checks. |
| Security | Allow-list file types, cap size, sanitize filenames, escape rendered text and CSV formula prefixes. | Malicious upload test suite. |
| Accessibility | Keyboard operation, semantic labels, visible focus, AA contrast, non-color risk cues. | Automated scan plus manual keyboard pass. |
| Compatibility | Latest Chrome, Edge and Firefox; responsive widths 360, 768 and 1440 px. | Browser/viewport matrix. |
| Observability | Structured logs include trace ID, event, duration, versions and error code; never raw uploaded rows. | Log assertions on success/failure. |
| Maintainability | Prediction and feature transforms share one pipeline; constants live in config; modules have unit tests. | Code review checklist. |

### Failure hierarchy

**Blocking**

No trustworthy output possible. Stop flow and name the corrective action.

**Degraded**

Output is usable with a limitation. Show persistent warning in UI and report.

**Advisory**

User can continue. Explain impact without interrupting the task.

**No fabricated fallback:** if model inference fails, WasteLens may show the last successful run with its original timestamp, but must not invent or silently substitute a fresh forecast.

---

## Cross-feature edge cases

| Area | Scenario | Expected result |
| --- | --- | --- |
| Input | Whitespace, mixed case or duplicate item aliases | Normalize approved aliases; preview mapping; never merge unknowns silently. |
| Input | Decimal comma, missing unit or impossible date | Block affected row and provide exact field guidance. |
| Behavior | Refresh/back during a completed run | Restore last committed run or show a clear expired-session state. |
| Behavior | Double click on Generate or Export | One operation; subsequent clicks disabled or deduplicated. |
| Data | Zero waste for all history | Allow training only if valid; disclose constant target and weak model value. |
| Data | New item with no historical rows | Category/default ratio fallback or “Insufficient item history”; no false precision. |
| Data | Future-dated rows in history | Reject and report rows. |
| Model | Model artifact/version mismatch | Block inference; log version error; no fallback to incompatible artifact. |
| Model | NaN, infinity or implausible output | Reject result, mark failed, retain inputs and trace ID. |
| Connectivity | Connection drops after request reaches server | Retry with same idempotency key; recover existing result if completed. |
| Third party | Weather provider timeout or quota limit | Use manual value or training median; label imputation. Core prediction remains available. |
| Export | Unicode item names or CSV formula prefix | Preserve text and escape dangerous leading characters. |
| Responsive | 320 px viewport or 200% zoom | No hidden actions or horizontal page scroll; tables convert to cards. |
| Concurrency | Dataset replaced while result tab is open | Mark result stale; block scenario promotion until regeneration. |

### Error message pattern

**What happened + what the user can do + stable reference.** Example: “Prediction could not be completed. Your inputs are saved; retry in a moment. Reference: WL-MODEL-503.” Do not expose exceptions, paths or raw payloads.

---

## QA strategy

### Automated coverage

- **Unit:** validators, feature transformations, risk thresholds, rounding, deltas and error mapping.
- **Model:** schema, deterministic seed, no-leak split, non-negative post-processing, baseline comparison.
- **Integration:** import -> predict -> recommend -> scenario -> export using golden fixture.
- **Contract:** required fields, nullable behavior, stable codes, units and version fields.
- **UI smoke:** controls, loading, validation, retry, responsive navigation and download.

### Manual focus

- Chart legibility and truthful axes.
- Plain-language explanations.
- Keyboard and screen-reader sequence.
- Mobile layout at 360 px and 200% zoom.
- Model warnings visible before acceptance.

### Priority acceptance matrix

| ID | Scenario | Pass condition |
| --- | --- | --- |
| P0-01 | Golden CSV | Imports and enables plan |
| P0-02 | Missing column | Blocked with exact name |
| P0-03 | Generate twice | One versioned run |
| P0-04 | Prediction failure | No fresh fake values |
| P0-05 | Recommendation math | Matches approved formula |
| P0-06 | Scenario reset | Baseline restored exactly |
| P1-01 | SHAP fails | Degraded explanation state |
| P1-02 | Mobile 360 px | All critical actions usable |
| P1-03 | PDF failure | CSV fallback offered |

### Definition of done

All P0 tests pass; no open critical/high defects; model beats or is honestly compared with naive baseline; Must flow works from a clean setup; error states preserve valid work; accessibility smoke passes; README states setup, data schema, model limits and demo steps; presentation screenshots use the released build.

**Test data:** maintain four fixtures: golden valid, 20% invalid rows, sparse 14-day history, and adversarial values. Keep expected outputs versioned with the model artifact.

---

## Success metrics

| Metric | Definition | MVP target | Measurement |
| --- | --- | --- | --- |
| Plan completion | Successful plan / started plan sessions | >=90% in demo tests | Funnel events |
| Time to recommendation | Data ready to plan displayed | <2 min median | Client timestamps |
| Prediction latency | Request to complete response | <5 s p95 | Server duration |
| Demand MAE | Absolute forecast error in meals | Beat naive baseline | Chronological holdout |
| Waste MAE | Absolute error in kg/item | Beat naive baseline | Chronological holdout |
| Recommendation adoption | Items accepted unchanged / shown | Observe; no fake target | Accept/override events |
| Estimated waste avoided | Baseline planned minus final planned, bounded at zero | Report transparently | Per run |
| Actual waste reduction | Waste rate change vs comparable baseline | Pilot KPI, post-MVP | Operational logs |
| Usability | Task success and 5-point ease score | >=80%; >=4/5 | Five-user test |
| Trust | “I understand this recommendation” | >=4/5 | Post-task survey |

### Event instrumentation

```text
dataset_upload_started | dataset_validation_completed
plan_generation_started | plan_generation_completed | plan_generation_failed
recommendation_overridden | scenario_applied | scenario_promoted
explanation_viewed | report_downloaded
```

Common properties: session_id, dataset_id, run_id, model_version, screen_size_class, duration_ms, status, error_code. Exclude raw menu rows and personal information.

**Interpretation:** “estimated waste avoided” is a decision-model estimate, not verified environmental impact. Verified reduction requires actual post-service waste data and a comparable baseline period.

---

## Seven-day delivery plan

| Day | Build focus | Exit gate |
| --- | --- | --- |
| **1** | Freeze scope; define schema; prepare real/synthetic-labeled dataset; wireframe mobile and desktop. | Golden CSV validates; approved user flow and data dictionary. |
| **2** | Exploratory analysis; unit normalization; feature engineering; naive baselines. | Reproducible notebook/script; leakage review complete. |
| **3** | Train demand and waste models; chronological evaluation; save pipeline and metrics. | Versioned artifact loads and produces valid outputs. |
| **4** | Recommendation rules, thresholds, prediction service/module, automated calculation tests. | Import-to-recommend path works without polished UI. |
| **5** | Responsive Streamlit dashboard; empty/loading/error states; item plan and risk views. | Must flow works at 360, 768 and 1440 px. |
| **6** | What-if, explanation, trends and export in priority order; performance pass. | Should features pass smoke or are cleanly feature-flagged off. |
| **7** | Regression, accessibility, unusual inputs, README, screenshots and 3-minute demo rehearsal. | All P0 tests pass; no new features; release tagged. |

### Ownership split for a small team

**Data / ML**

Schema, analysis, features, baselines, trained artifact, SHAP and metrics.

**App / UI**

Streamlit flow, responsive layout, state management, charts and export.

**QA / product**

Fixtures, acceptance cases, content, accessibility, demo and submission assets.

### Cut order if behind

Cut model comparison UI first, then PDF export, then local SHAP, then historical filter richness. Preserve valid import, prediction, item recommendations, error honesty and responsive core flow.

**Release rule:** by the end of Day 5 there must be a complete, presentable Must path. Days 6-7 improve and prove it; they do not rescue the architecture.

---

## Risks, open decisions and glossary

### Top risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Insufficient real data | Label synthetic rows; separate real vs synthetic evaluation. |
| Model underperforms baseline | Show honest comparison; retain rules-led recommendation demo. |
| Streamlit CSS breaks | Progressive enhancement; native controls stay usable. |
| Scope exceeds seven days | Must/Should/Could gates and fixed cut order. |
| False precision | Confidence labels, units, rounding and limitations. |
| Demo environment fails | Pin dependencies; local assets; rehearsal from clean setup. |

### Open product decisions

- Which kitchen segment and meal period will the demo represent?
- What serving ratios and production increments are authoritative?
- What data is real, synthetic, or imputed?
- Is optional weather worth the reliability cost?
- Which achieved model metrics can be stated in the presentation?

### Glossary

**Baseline forecast:** original prediction used for comparison.

**Confidence label:** qualitative reliability indicator based on model error, data quality and input range.

**Demand:** meals expected to be served/consumed.

**Item waste rate:** predicted waste quantity divided by planned preparation quantity.

**MAE:** mean absolute error; average absolute prediction error.

**Recommendation:** rules-adjusted preparation quantity derived from forecast and operational settings.

**Scenario:** non-destructive recalculation using changed future inputs.

**SHAP:** method for estimating each feature's influence on a model output.

**Waste:** edible prepared food not served/consumed, measured in kilograms for MVP.

**Final product promise:** WasteLens gives a busy kitchen professional a faster, explainable and testable preparation decision while remaining clear about data quality and model limitations.

---
