# AGENTS.md — working on ha-adaptive-comfort

Guide for AI agents (and humans in a hurry) contributing to this codebase.

## What this is

A Home Assistant custom integration (`custom_components/adaptive_comfort/`) that acts as a
self-learning whole-home thermostat for multi-split AC systems. It learns each room's thermal
dynamics (1R1C model with outdoor + house-mixing coupling), disaggregates AC power from a
whole-house meter, and coordinates AC heads for comfort and efficiency.

## Architecture — the one invariant that matters

**`core/` is pure Python. Everything else is the HA adapter layer.**

- `core/` imports nothing from Home Assistant and must stay that way. All physics, estimation,
  and decision logic lives here so it can be unit-tested without an HA harness.
- `coordinator.py` (the runtime) is the only substantial HA-side module: it polls sensors,
  builds a `HouseSnapshot` every 60 s, calls `controller.tick(snapshot, state)`, and executes
  the returned `Command`s against real `climate` entities.
- Entity platforms (`sensor.py`, `switch.py`, `number.py`, `climate.py`, `binary_sensor.py`)
  are thin views over the runtime's state; they contain no logic.

If you find yourself importing `homeassistant.*` inside `core/`, you are in the wrong layer.

## Map

| Path | What lives there |
|---|---|
| `core/controller.py` | Pure decision engine: one tick from `HouseSnapshot` → `Decision` (mode arbitration, demand/helper selection, min-on/off guards, tracking-delta adaptation, shedding, fan assist, window suggestions). All tunable constants at the top. |
| `core/thermal.py` | Per-zone 1R1C `ThermalModel`: RLS fits of k_out / k_mix per door regime, effective capacitance, diurnal disturbance, sensible power, COP estimation. Outdoor anchoring via the air-exchange hypothesis (moisture k → absolute UA). |
| `core/power.py` | Load composition (grid + battery − known loads), time-of-day `BaselineModel`, step-delta AC estimation, per-zone power allocation, shedding math, `outdoor_band()` for COP bookkeeping. |
| `core/park.py` | Learned above-setpoint ("parked") head behavior: EWMA of extraction while parked + idle/trickle classification. Fed by the runtime from the thermal model's sensible power while the controller holds a zone parked. |
| `core/drift.py` | Per-head, per-operating-state internal-sensor offset learning (internal vs external reference). Used to correct readings and as *fallback* setpoint translation. |
| `core/comfort.py` | Adaptive comfort band (running-mean outdoor → band center/edges). |
| `core/types.py` | All dataclasses: `Settings`, `ZoneSnapshot`, `HouseSnapshot`, `Command`, `ControllerState` (+ its persistence round-trip). |
| `core/rls.py`, `core/series.py`, `core/psychro.py`, `core/simulator.py` | Recursive least squares, time-series ring buffer, psychrometrics, and a small sim house used by tests. |
| `coordinator.py` | Runtime: sensor ingestion, estimator updates (`_update_estimators`, `_update_zone_estimators`), COP tables, persistence (`_persist`/restore), command execution (incl. tracking translation), diagnostics dump. |
| `tests/` | Pytest suite. Core tests run without HA (see `conftest.py` stubbing); config-flow tests need `pytest-homeassistant-custom-component`. |

## Key semantics and unit conventions (violating these breaks physics silently)

- **Temperatures are °C; deltas are K. Power is W; energy is Wh (capacitance `c_eff_wh_per_k`).**
- **Two coordinate frames for setpoints.** The controller thinks in *room frame* (external
  sensor). Heads regulate on their *internal* sensor, which reads ~3 K low while cooling
  (supply-air contamination). Translation between frames happens **only** in
  `coordinator.py` command execution — either dynamically (tracking control: command
  `internal − delta`, re-anchored every `COMMAND_SPACING_S`) or statically via
  `DriftEstimator.offset(state)` as fallback. Never translate anywhere else.
- **`sensible_power_w` is signed**: negative while cooling. It includes the storage term
  `C·dT/dt`, which is why transient COP samples are valid.
- **Multi-split constraint**: all heads share one compressor mode. Opposite-demand zones can
  only get `fan_only` (fan assist), and only after `COIL_DRY_S` — running a fan over a wet
  coil re-evaporates condensate and undoes latent work already paid for.
- **Door regimes**: `ThermalModel.fits[door_open]` holds separate RLS fits. A regime with no
  samples *reports the other regime's fit* (`_fit` fallback); `fit_is_fallback()` tells you
  whether you're looking at learned or borrowed numbers. Diagnostics expose this flag —
  keep it honest.
- **COP tables**: `cop_table` is keyed by active head count; `cop_table_banded` by
  `"{heads}|{band}"` with bands from `power.outdoor_band()` (mild/warm/hot). Head count and
  weather are confounded in the field (3 heads ↔ hot afternoons), so never draw head-count
  conclusions from the unbanded table alone.
- **Persistence**: everything that must survive a restart goes through `coordinator._persist()`
  / restore and the `to_dict`/`from_dict` pairs. If you add controller state or a setting,
  wire all four places: dataclass, `to_dict`, `from_dict`, and the `_persist` settings dict
  (+ restore key list) — and the switch/number entity if user-facing.

- **Parking is measured, never assumed.** A zone leaving demand on a multi-split (siblings
  keeping the compressor alive) may be *parked* — setpoint `internal ± preferred_margin`
  (cool/heat), mode kept — instead of turned off: bounded probes (`PARK_PROBE_S`, spaced
  `PARK_PROBE_SPACING_S`) while behavior is unclassified, exploitation once a head is a
  known trickler whose learned output covers the zone's `standing_load_w`. Session margin
  starts one step below `preferred_margin_k` (floored at `PARK_MARGIN_K` so the setpoint
  stays on the park side of the internal reading), escalates while the room keeps moving
  in the conditioning direction (up to `PARK_MARGIN_MAX_K`), and settles the preferred
  depth on a clean exit. Devices differ (thermo-off vs keep-temperature trickle) and the
  estimator learns which; nothing hardcodes either answer. Shed zones never park. Helper
  selection outranks parking. The overcorrection release sits `PARK_OVERCOOL_BUFFER_K`
  *below* the band floor (cool; above the ceiling in heat) — zones exit demand AT the floor,
  so a guard placed on the floor itself kills every park at entry (a real field failure).
  Probe budget is charged at *release* and only when the probe produced observations;
  stillborn probes get the short `PARK_PROBE_RETRY_S` clock instead of the 6 h spacing.
- **Run-out is the free probe window.** A zone wanting off while `min_on` forces it to run
  is parked immediately — no sibling requirement, no probe budget — because the compressor
  is alive regardless: observations are free, and an idling head saves energy versus tracked
  run-out. Parked releases honor `min_on`; continued exploitation beyond it requires a live
  sibling (`want_on`); park direction follows `head_mode` when the house's dominant mode goes
  idle; overcorrected-but-unstoppable zones hold at `PARK_MARGIN_MAX_K` rather than being
  released into a forbidden off. Tracked `runout` commands are the `park_learning`-off
  fallback only.
- **Power is the only honest activity signal.** These heads report `hvac_action: cooling`
  through entire parks while the electrical record is bimodal (fan-only <60 W vs real
  compression >300 W) — the device cycles via an internal hysteresis around its own sensor.
  Parked observations are therefore power-gated (`park.gate_observation`): solo-park draw
  below `FAN_FLOOR_W` forces extraction to zero. Never infer compression from `hvac_action`.
  Duty samples run on the 60 s control tick (not the 5 min thermal fit), with
  `PowerDebounce` / `PARK_DUTY_DEBOUNCE_S` so brief meter blips do not flip
  `active_ratio`. The per-margin `margin_bins` map (extraction, compression duty, n)
  charts the hysteresis; `coast_margin_k()` gives the cheapest coasting depth;
  `cop_table_state` buckets house COP by control state to audit park-hold economics.
- **Regime policy owns the macro decision.** `_select_regime` picks per tick (15 min dwell):
  `ventilate` when outdoor beats the coolest target by `REGIME_VENT_MARGIN_K` (forces the
  grace-gated window flow — never hard-strands a hot room), `continuous` when aggregate
  `standing_load_w` can feed `REGIME_CONT_LOAD_FRACTION` of the compressor's thermal floor
  (zones park instead of turning off; sibling requirement waived; the head's measured
  hysteresis does fine modulation), else `cycling` (true offs, drained coils). `auto_regime`
  off restores pre-regime semantics exactly. Heat shares the `continuous` path when load can
  feed the floor; `ventilate` remains cool-only. Opt-in `night_ventilate` (22:00–08:00) widens
  the ventilate margin to 0 K and suppresses continuous when outdoor is within
  `NIGHT_SKIP_CONT_K` of the coolest target. Tune thresholds from `cop_table_state` and
  the `StartCounter` diagnostics, not from intuition. Park entry prefers
  `coast_margin_k()` once the hysteresis map has found a coasting depth.
- **Zone COP counts latent.** `update_cop` heat flow is sensible + latent; sensible-only
  samples in humid rooms undercount delivered cooling by 30–50% and can fall below
  `COP_MIN`, producing absurd or silently-rejected readings.
- **Control/park history is entity-backed.** Zone `control_state` is the single role
  timeline (`off`/`demand`/`helper`/`park`/`runout`/…); its attributes carry
  `last_reason`, park margins/classification, and track depth — so separate
  `command_reason` / `zone_parked` / `park_preferred_margin` entities are not needed.
  Chart the continuous internals with `head_internal_temp`, `track_delta`,
  `park_margin`, `park_extraction`, and `park_classification` (plus house
  `parked_zones`, `operating_regime`, `compressor_starts_per_hour`).

## Control-loop cheatsheet (what happens each 60 s tick)

1. Runtime ingests sensors → corrected zone temps (drift-aware), power composition, AC
   estimate (baseline + step deltas), zone estimator updates (free-float fits when off;
   COP/c_eff when conditioning).
2. `controller.tick`: pick dominant mode (model-driven free-float prediction), select demand
   + helper zones, apply min-on/min-off and mode-change rate guards, adapt per-zone tracking
   delta, apply shedding/window/fan-assist policy, emit `Command`s.
3. Runtime executes: translates each command to per-head device setpoints (tracking or drift
   frame), fans out to mirrored heads, records transitions.

## Testing & tooling

```bash
pip install pytest ruff
pytest tests -q                          # core logic; config-flow tests need the HA harness
pip install -r requirements_test.txt     # adds pytest-homeassistant-custom-component
pytest tests -q
ruff check . && ruff format .
```

- Core tests build zones with the `make_zone`/`make_snapshot`/`warmed_state` helpers
  (see `tests/test_controller.py`); reuse them rather than hand-rolling snapshots.
- `warmed_state` matters: fresh `ControllerState` timers block transitions via min-off guards
  and produce confusing "nothing happened" tests.
- Physics tests use `core/simulator.py` (`SimHouse`/`SimRoom`) to generate consistent
  trajectories; prefer it over synthetic constants when testing estimators.
- CI: ruff, hassfest, HACS validation, pytest on Python 3.13.

## Gotchas learned the hard way

- Commands are fire-and-forget (`blocking=False`); heads may unilaterally stop conditioning
  when their internal sensor crosses the device setpoint. Tracking control exists precisely
  because a static drift translation over-demands at run start and under-demands at run end
  (visible as sub-`min_on` hvac_action bursts in field data).
- `p_ac` near cycle edges divides small numbers: gate COP publication on
  `HOUSE_COP_MIN_POWER_W` and smooth with `HOUSE_COP_EMA_ALPHA`.
- The house may have zones with multiple mirrored heads and a single sensor
  (`n_rooms > 1`): per-head power is `allocated_w / n_rooms`, and thermal totals multiply
  back by `n_rooms`. Keep the two consistent. Park learning stays in the per-head
  frame (`park_extraction_w`, trickle thresholds); `standing_load_w` is zone-total, so
  exploit compares extraction to `standing_load_w / n_rooms`. Solo-park electrical
  gating uses `fan_floor_w(n_rooms)` so multi-head fan draw is not mistaken for compression.
- Timezone: runtime uses `local_hour` for diurnal models; timestamps are epoch seconds.
