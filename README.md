# FlowSight — Urban Traffic Flow & Incident Intelligence

> An AI decision-support system that infers what is happening on a city road network, anticipates what happens next, recommends simulated diversions and traffic-management responses, and identifies where infrastructure changes would cut recurring congestion — with evidence, uncertainty and expected impact attached to every recommendation.

**Hackathon:** Neurax Hackathon 3.0 · **Domain 1:** AI in Smart Cities
**Problem statement:** Urban Traffic Flow & Incident Intelligence
**Team:** Databaes — J Surya, D Shriya, G Nihal
**Status:** Checkpoint 1 (design and approach). Implementation status is tracked in [Roadmap](#8-roadmap).

---

## Contents

1. [Problem Understanding](#1-problem-understanding)
2. [Requirements → Solution Map](#2-requirements--solution-map)
3. [System Architecture](#3-system-architecture)
4. [Approach](#4-approach)
5. [Explainability, Confidence and Robustness](#5-explainability-confidence-and-robustness)
6. [Evaluation Plan](#6-evaluation-plan)
7. [Assumptions, Limitations and Risks](#7-assumptions-limitations-and-risks)
8. [Roadmap](#8-roadmap)
9. [Planned Repository Layout and Setup](#9-planned-repository-layout-and-setup)

---

## 1. Problem Understanding

### 1.1 The operating environment

The challenge is set in a **Hyderabad-like** urban network, which combines several conditions that make traffic management hard:

- **Dense mixed traffic** (cars, two-wheelers, autos, buses, freight) with strong **peak-hour commuter flows**.
- **Signalized junctions, flyovers and arterial corridors**, where a problem at one node propagates along the corridor.
- **Recurring bottlenecks** caused by fixed geometry or capacity limits (merges, lane drops, signal-limited junctions).
- **Non-recurring disruptions**: incidents, road works, weather-related slowdowns, and event-driven surges.
- **Congestion spillback**: a queue that grows past its segment and blocks upstream or neighbouring segments.

Conditions can change within minutes, but a fraction of the congestion is *structural* and returns every day. A useful system has to treat these two very differently.

### 1.2 What we are being asked to build

A **software-only decision-support system**, not a navigation app and not a generic chatbot. Working from organizer-provided traffic and road-network datasets, it must:

| # | Capability | Meaning |
|---|---|---|
| 1 | **Infer** | Maintain a continuously updated view of network conditions; identify congestion and abnormal behaviour |
| 2 | **Detect** | Detect or classify incidents where the data supports it, with controlled false alarms |
| 3 | **Anticipate** | Forecast traffic state **15–60 minutes ahead** |
| 4 | **Recommend (operational)** | Generate evidence-based diversion and traffic-management advisories |
| 5 | **Recommend (structural)** | Propose data-driven road-network or infrastructure changes for recurring bottlenecks, with estimated before/after impact |

### 1.3 Hard constraints

- All actions, diversion plans and construction suggestions are **simulated or advisory only**.
- **No** live signal control, camera access, GPS-device integration, roadside-sensor integration, municipal infrastructure access or real construction.
- Only **organizer-provided datasets** are analysed.

### 1.4 Who the system is for

The primary user is a **traffic control-room operator or planner**, not a commuter. This shapes the design:

- The operator needs to know **what happened, where, how sure the system is, and what to do**, within seconds.
- A false alarm costs operator attention and trust, so **precision and confidence reporting matter as much as recall**.
- A planner needs to know **which bottlenecks are worth fixing first** and what the fix is likely to achieve, including the uncertainty.

### 1.5 Key insight that drives the design

> **Recurring congestion and non-recurring congestion need different responses.**
> A predictable peak-hour queue at a known bottleneck calls for scheduling and structural fixes. An abrupt, unexpected slowdown calls for incident response and diversion. Separating the two is the foundation of every module below.

### 1.6 What success looks like

- Congestion and incidents are detected **correctly with few false alarms**.
- Forecasts at 15/30/45/60 minutes **beat naive baselines on unseen conditions**.
- Recommendations are **relevant, feasible, explained and quantified**.
- Performance **degrades gracefully** under changed demand and noisy or missing data.
- Every output carries **evidence and a confidence level**, and the system says so when it does not know.

---

## 2. Requirements → Solution Map

| Requirement | Our module | Key technique | Output |
|---|---|---|---|
| Continuously updated network view | Replay engine + state estimator | Simulated clock streams data; per-segment congestion index | Live network state map |
| Congestion and abnormal behaviour | Congestion & anomaly detector | Baseline-relative scoring (median/MAD, CUSUM) | Congestion level + anomaly score per segment |
| Incident detection / classification | Incident detector & cause classifier | Shockwave signature + spatial agreement + persistence rules | Incident alerts with type and confidence |
| Forecast 15–60 min | Forecasting engine | Gradient-boosted models per horizon, graph-neighbour features, quantile intervals | Forecast state with uncertainty band |
| Diversion / management advisories | Advisory engine | Forecast-aware k-shortest alternates with capacity guard | Advisory cards with expected benefit and side-effects |
| Infrastructure suggestions | Bottleneck & infrastructure planner | Delay-hours ranking, bottleneck diagnosis, intervention catalogue | Ranked proposals with before/after estimates |
| Before/after impact | What-if simulator | BPR-style volume-delay model with demand uncertainty | Impact ranges, not single numbers |
| Explainability and confidence | Evidence and confidence layer | Rule traces, feature attribution, calibrated intervals, abstention | Evidence panel per alert |
| Robustness to unseen patterns | Stress-test harness | Data dropout, noise, demand shifts, held-out scenarios | Degradation curves |

---

## 3. System Architecture

![FlowSight architecture](docs/architecture.svg)

### 3.1 Layers and data flow

1. **Data layer:** Organizer-provided datasets (traffic observations, road-network description, and weather/event/roadwork information where supplied).
2. **Ingestion and schema adapter:** A config-driven adapter maps whatever column names and units the dataset uses into one internal schema. Cleaning, gap detection and imputation run here. A **replay clock** streams historical data in time order so the system behaves as if it were live.
3. **Road-network graph and feature store:**
   - The **graph** holds segments and junctions with free-flow speed, capacity, lane count and connectivity. It is what makes spillback and diversions computable.
   - The **feature store** holds time-of-day and day-of-week baselines, lag features, rolling statistics and neighbour-segment aggregates.
4. **Analytics core:** three modules run on every time step:
   - Congestion and anomaly detection
   - Incident detection and cause classification
   - Forecasting for 15, 30, 45 and 60 minutes with prediction intervals
5. **Decision layer:**
   - **Advisory engine:** diversion and traffic-management recommendations.
   - **Bottleneck and infrastructure planner:** ranks recurring bottlenecks and proposes network changes.
   - **What-if simulator:** estimates the before/after effect of any advisory or proposal.
6. **Operations dashboard:** map, time slider, alert feed, evidence panels, recommendations and a what-if console.
7. **Cross-cutting layers:** explainability and confidence handling, robustness and data-quality checks, and an evaluation harness.

### 3.2 Design principles

- **Schema-agnostic ingestion.** The adapter isolates dataset quirks so the core never changes when the data format does.
- **Replay-driven "live" behaviour.** No real-time integrations, but the system is evaluated as a streaming system.
- **Uncertainty-first.** Every forecast has an interval, and every advisory has a confidence value. Below a confidence threshold the system reports "monitor only" instead of recommending.
- **Advisory-only by construction.** No module has a code path that touches external systems; all actions are simulated objects.
- **Baselines everywhere.** Each learned component is reported against a simple baseline, so improvements are demonstrable.
- **Reproducible.** Fixed seeds, pinned dependencies, a single command to run the pipeline.

### 3.3 Planned technology stack

| Concern | Choice |
|---|---|
| Language | Python 3.11 |
| Data handling | pandas / polars, NumPy |
| Graph and routing | NetworkX (OSMnx only if the organizers permit external static road data) |
| Forecasting | LightGBM (quantile objectives), scikit-learn |
| Explainability | SHAP for forecasters; rule traces for detectors |
| Simulation | Custom BPR-style volume-delay model with iterative assignment |
| Backend | FastAPI |
| Dashboard | Streamlit + pydeck/Folium (or React + MapLibre if frontend capacity allows) |
| Reproducibility | requirements file / Docker, Makefile, pytest |

---

## 4. Approach

### 4.1 Congestion and anomaly detection

- **Congestion index** per segment per time step: speed ÷ free-flow speed (or a travel-time index if speed is not available), mapped to levels (free-flow, moderate, heavy, severe).
- **Baselines** by segment, time of day and day of week, so that "slow" is judged against what is normal for that place and time.
- **Anomaly score** = deviation from baseline using robust statistics (median/MAD) and CUSUM for sustained shifts.
- **Recurring vs non-recurring label:** congestion that matches the baseline is *recurring*; congestion that exceeds it is *non-recurring*. This label feeds the advisory and planning modules.
- **Congestion clusters:** connected congested segments on the graph are grouped, and the **downstream head of the cluster** is identified as the likely root bottleneck. This captures spillback across neighbouring segments.

### 4.2 Incident detection and classification

Incidents are detected from the *pattern*, not from a single slow reading:

- **Signature:** abrupt speed drop at a segment, growing slowdown on upstream segments, and reduced flow downstream.
- **False-alarm control:** minimum persistence window, hysteresis between raise and clear thresholds, and agreement from neighbouring segments.
- **Cause classification** using segment-level features and context flags:

| Class | Typical evidence |
|---|---|
| Incident | Abrupt, localized onset with upstream queue growth |
| Event-driven surge | Gradual, area-wide rise around a known venue or time |
| Weather slowdown | Wide-area slowdown coinciding with a weather flag |
| Road works | Persistent, localized reduction, often scheduled |
| Recurring bottleneck | Matches the baseline at the same place and time |

- Where the dataset provides no labels, we will **inject synthetic incidents** into held-out data to measure detection rate, detection delay and false alarms per segment-day.
- Classes the data cannot support are reported as "unclassified slowdown" rather than guessed.

### 4.3 Traffic forecasting (15/30/45/60 minutes)

- **Model:** gradient-boosted trees, one model per horizon, on lag features, rolling statistics, time features, neighbour-segment aggregates, and weather/event flags where available.
- **Uncertainty:** quantile models produce prediction intervals, which drive confidence handling downstream.
- **Baselines:** persistence (last value) and historical average by time of day. The learned model must beat both at every horizon.
- **Validation:** strictly **time-based splits** (no random shuffling), plus **held-out days or scenarios** (for example a rainy day or an event day) to measure performance on unseen conditions.
- **Scope choice:** we start with the tree-based approach because it is fast to train, easy to explain and robust to missing inputs. A graph neural network is a stretch goal only if it demonstrably beats this model.

### 4.4 Adaptive recommendations: diversions and traffic-management advisories

For each significant non-recurring event or forecast congestion cluster:

1. Locate the head of the congestion and the upstream decision points.
2. Generate **k alternative routes** on the graph using **forecast travel times**, not current ones.
3. Apply a **capacity guard**: reject or down-weight alternates that are already near capacity or forecast to congest, and divert only a fraction of the flow, so the diversion does not simply move the problem.
4. Choose complementary **management actions** (all simulated): signal green-time extension using a Webster-style split, marshal or police deployment points, and variable-message-sign text.
5. Estimate benefit with the what-if simulator and attach: action, location, timing, expected delay saved (range), side-effects, confidence, and evidence.

If confidence is below threshold, the system issues a **watch** notice instead of a recommendation.

### 4.5 Infrastructure and network modification proposals

- **Identify recurring bottlenecks:** segments congested in a high share of comparable peak windows.
- **Rank by impact:** *delay-hours per day* = Σ (travel time − free-flow travel time) × flow.
- **Diagnose the type:** merge or lane drop, signal-limited junction, geometry-limited stretch, or spillback victim (congested because of something downstream).
- **Match to an intervention catalogue:** signal retiming, extra lane or slip road, turn restriction, one-way pairing, bus priority, and similar measures. Each entry has a capacity effect and a relative cost tier.
- **Estimate before/after:** modify the affected segment's capacity in the simulator, re-run the flows, and report **ranges** using demand-uncertainty sampling.
- **Check for bottleneck shifting:** the simulator verifies that the fix does not just move the queue to the next segment downstream.

### 4.6 What-if simulator

A lightweight macroscopic model: segment travel time follows a **BPR volume-delay function**, `t = t0 · (1 + α · (v/c)^β)`, with parameters calibrated against the observed data. Flows are re-assigned across alternate routes by iterative averaging. It is deliberately simple: fast enough to run interactively in the dashboard and transparent enough to explain. It is an *estimator of relative benefit*, not a microscopic traffic simulation, and we state this in every impact figure.

### 4.7 Dashboard

- **Map** with segments coloured by current state, and a **time slider** for now, +15, +30, +45 and +60 minutes.
- **Alert feed** with severity, type and confidence.
- **Evidence drawer** for each alert (see next section).
- **Recommendation panel** with expected before/after impact.
- **Bottleneck leaderboard** for planners.
- **What-if console:** close a road, scale demand, and see the simulated effect.

---

## 5. Explainability, Confidence and Robustness

### 5.1 Explainability

Every alert or recommendation includes an **evidence panel**:

- Observed vs baseline chart for the affected segment
- State of neighbouring segments
- The rules that fired (detectors) or top contributing features (forecasters)
- The forecast and its interval
- Known limitations for that specific alert

### 5.2 Confidence handling

- **Forecast confidence** from calibrated prediction-interval width.
- **Data-quality score** from completeness and recency of inputs to the alert.
- **Detector agreement** across independent signals (speed drop, upstream growth, flow drop).
- A composite confidence value gates actions: **recommend / watch / abstain**.
- We check **calibration**: for example, that 80% intervals cover roughly 80% of outcomes on held-out data.

### 5.3 Robustness to unseen patterns

A stress-test harness re-runs the evaluation under controlled perturbations:

| Perturbation | Levels |
|---|---|
| Missing data (sensor dropout) | 10%, 20%, 30% |
| Measurement noise | low, medium, high |
| Demand shift | −20%, +20%, +40% |
| Held-out scenario | rain day, event day, road-works day (where present in data) |
| Synthetic disruption | injected incident or closure |

We report how accuracy, detection rate and false alarms **degrade**, rather than only the best-case result.

---

## 6. Evaluation Plan

| Judging criterion | What we measure |
|---|---|
| Congestion and incident detection | Precision, recall, F1; false alarms per segment-day; detection delay; performance on injected incidents |
| Forecasting accuracy | MAE/RMSE of speed or travel time and F1 of congestion level, per horizon (15/30/45/60), on held-out days; comparison to persistence and historical-average baselines |
| Recommendation quality | Simulated delay reduction (vehicle-hours); feasibility checks passed (capacity guard, connectivity); side-effects reported |
| Robustness | Degradation curves under the perturbations in §5.3 |
| Explainability and confidence | Interval calibration; evidence completeness; abstention rate and accuracy when abstaining |
| Technical implementation | One-command reproducibility, pinned dependencies, unit tests for core logic |
| UI/UX | Time from alert to understood recommendation; clarity of map, alerts and forecast views |
| Innovation | Ablations showing that recurring/non-recurring separation, forecast-aware diversions and confidence gating each improve decisions |

**Results table (to be filled as experiments complete):**

| Horizon | Persistence MAE | Hist. avg MAE | FlowSight MAE | Interval coverage |
|---|---|---|---|---|
| 15 min | TBD | TBD | TBD | TBD |
| 30 min | TBD | TBD | TBD | TBD |
| 45 min | TBD | TBD | TBD | TBD |
| 60 min | TBD | TBD | TBD | TBD |

---

## 7. Assumptions, Limitations and Risks

**Assumptions**

- The organizer datasets contain, at minimum, a timestamped speed, flow or travel-time signal per road segment, plus enough network information to build a connected graph. The adapter is designed to tolerate differences in schema.
- Weather, event and roadwork information is used only if it is in the provided data.

**Limitations (stated openly)**

- The what-if simulator is macroscopic. Its impact figures are **estimates of relative benefit with ranges**, not guarantees.
- Incident *cause* classification is limited by what the data records. Without labels, we validate with synthetic incidents, which is weaker evidence than real labelled events.
- Forecast accuracy on entirely novel situations (for example, a citywide disruption never seen in training) will be lower, and the system reports wider intervals in that case.
- All recommendations are advisory and would need validation by traffic engineers before any real-world use.

**Risks and mitigations**

| Risk | Mitigation |
|---|---|
| Dataset schema differs from our assumption | Config-driven adapter; isolate schema handling in one module |
| Too few labelled incidents | Synthetic injection plus rule-based detection with conservative thresholds |
| Deep models overfit or fail to beat baselines | Start with tree models; keep baselines in every report |
| Diversions shift congestion elsewhere | Capacity guard and bottleneck-shift check in the simulator |
| Time pressure | Thin end-to-end slice first, then improve modules in order of judging weight |

---

## 8. Roadmap

| Checkpoint | Goal | Deliverables |
|---|---|---|
| **1 — README (15)** | Problem understanding, architecture, approach | This document and the architecture diagram |
| **2 — Partial execution (25)** | Working end-to-end thin slice | Ingestion + graph → congestion map → baseline forecast → one working advisory, shown in the dashboard |
| **3 — Full evaluation (60)** | Complete, measured system | Incident detection, tuned forecasters with intervals, capacity-aware diversions, bottleneck planner with before/after, robustness harness, calibrated confidence, polished UI |

Current status: **Checkpoint 1 — design complete.**

---

## 9. Planned Repository Layout and Setup

```text
flowsight/
├── README.md
├── docs/
│   └── architecture.svg
├── configs/
│   └── schema.yaml          # dataset column mapping and thresholds
├── data/                    # organizer datasets (not committed)
├── src/
│   ├── ingest/              # adapter, cleaning, imputation, replay clock
│   ├── graph/               # road-network graph and routing
│   ├── detect/              # congestion, anomaly, incident, cause classifier
│   ├── forecast/            # features, models, baselines, intervals
│   ├── advise/              # diversion and management advisories
│   ├── plan/                # bottleneck ranking, infrastructure proposals
│   ├── simulate/            # BPR-style what-if simulator
│   ├── explain/             # evidence panels, confidence, calibration
│   └── eval/                # metrics, stress tests, synthetic incidents
├── app/                     # dashboard
├── tests/
├── requirements.txt
└── Makefile
```

**Planned commands** (to be finalised at Checkpoint 2):

```bash
pip install -r requirements.txt
make data      # ingest and clean organizer datasets
make train     # build baselines and train forecasters
make eval      # run metrics and stress tests
make app       # launch the dashboard
```

---

*All outputs of this system are simulated and advisory. No live infrastructure, sensors, cameras or signals are accessed.*
