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
| `core/thermal.py` | Per-zone 1R1C `ThermalModel`: RLS fits of k_out / k_mix per door regime, effective capacitance, diurnal disturbance, sensible power, COP estimation. Power is fit-consistent (`UA = c_eff·k`, `Q_ac = c_eff·(dT/dt − free_float_rate)`); outdoor *volume* flow for moisture stays `k_out·V`. |
| `core/power.py` | Load composition (grid + battery − known loads), time-of-day `BaselineModel`, step-delta AC estimation, per-zone power allocation, shedding math, `outdoor_band()` for COP bookkeeping. |
| `core/park.py` | Learned above-setpoint ("parked") head behavior. Two axes: (1) thermal class from extraction EWMA — `idle` ≤25 W heat removed, `residual` ≥60 W residual cooling, else `unknown` (not fan electrical watts); (2) electrical fan floor / duty — `p_ac` below `fan_floor_w(N)` is fan-type park (no compression). Fed by the runtime from the thermal model's sensible power while parked. |
| `core/drift.py` | Per-head, per-operating-state internal-sensor offset learning (internal vs external reference). Used to correct readings and as *fallback* setpoint translation. |
| `core/comfort.py` | Comfort band, presets (`effective_preset`), demand setpoints (center vs Away/vacant band-hold), `effective_zone_occupied` for zone presence gating, and model mode arbitration (`demand_integrals` over an 8 h free-float horizon with asymmetric exit). |
| `core/types.py` | All dataclasses: `Settings`, `ZoneSnapshot`, `HouseSnapshot`, `Command`, `ControllerState` (+ its persistence round-trip). |
| `core/rls.py`, `core/series.py`, `core/psychro.py`, `core/simulator.py` | Recursive least squares, time-series ring buffer, psychrometrics, and a small sim house used by tests. |
| `coordinator.py` | Runtime: sensor ingestion, estimator updates (`_update_estimators`, `_update_zone_estimators`), COP tables, persistence (`_persist`/restore), command execution (incl. tracking translation), diagnostics dump. Power-react is a lean path (heads+power+demand; full `controller.tick` only on shed engage/release). |
| `tests/` | Pytest suite. `conftest.py` loads `pytest-homeassistant-custom-component` for the whole suite (needed by config-flow / any HA tests); most `core/` tests do not exercise HA APIs. |

## Key semantics and unit conventions (violating these breaks physics silently)

- **Temperatures are °C; deltas are K. Power is W; energy is Wh (capacitance `c_eff_wh_per_k`).**
- **Two coordinate frames for setpoints.** The controller thinks in *room frame* (external
  sensor). Heads regulate on their *internal* sensor, which reads ~3 K low while cooling
  (supply-air contamination). Translation between frames happens **only** in
  `coordinator.py` command execution — either dynamically (signed head depth:
  cool `internal + depth`, heat `internal − depth`; negative = chase track,
  positive = hysteresis park) re-anchored every `COMMAND_SPACING_S` **or**
  sooner when `|device_SP − ideal| ≥ HEAD_REANCHOR_EPS_K` after
  `HEAD_REANCHOR_MIN_S` — or statically via `DriftEstimator.offset(state)` as
  fallback. Never translate anywhere else.
- **`sensible_power_w` is signed**: negative while cooling. It is
  `c_eff · (dT/dt − free_float_rate)` — same rates `predict_free` integrates —
  so standing load / COP / park extraction agree with the temperature model.
  Transient COP samples are valid because the storage term is inside that excess.
- **Multi-split constraint**: all heads share one compressor mode. Opposite-demand zones can
  only get `fan_only` (fan assist), and only after `COIL_DRY_S` — running a fan over a wet
  coil re-evaporates condensate and undoes latent work already paid for.
- **Door regimes**: `ThermalModel.fits[door_open]` holds separate RLS fits. A regime with no
  samples *reports the other regime's fit* (`_fit` fallback); `fit_is_fallback()` tells you
  whether you're looking at learned or borrowed numbers. Diagnostics expose this flag —
  keep it honest. **No door sensor → assume open** (inter-room mixing by default); on
  restore, closed-fit history migrates into open when open is still empty.
- **COP tables**: all ledgers are **mode-split** (`cool|…` / `heat|…`) so
  seasons never mix. `cop_table` keys `"{mode}|{heads}"`; `cop_table_banded`
  `"{mode}|{heads}|{band}"` with bands from `power.outdoor_band()` (mild/warm/hot);
  `cop_table_state` `"{mode}|{conditioning|park|mixed}"` from signed depth
  (`depth_k ≤ 0` → conditioning, `> 0` → park); `cop_table_depth`
  `"{mode}|{±0.5}"` learns chase / zero-hold / hysteresis COP on the continuum
  grid. Snapshot hints (`cop_by_head_count`, `cop_by_band`) filter to the
  active controller mode. Samples file under that same mode only when an active
  head also reports the matching `hvac_action` (`cop_sample_mode`) — mode-change
  lag skips the row rather than contaminating the other season. Head count and
  weather stay confounded inside a mode (3 heads ↔ hot afternoons) — never draw
  head-count conclusions from the unbanded table alone. Helper drop uses
  `cop_by_head_count_banded` (live outdoor band, demand vs demand+helpers) and
  falls back to unbanded only when that band lacks both N keys. Schema 2 bare keys
  migrate into `cool|…` on restore; schema 4 adds the depth ledger.
- **Persistence**: everything that must survive a restart goes through `coordinator._persist()`
  / restore and the `to_dict`/`from_dict` pairs. If you add controller state or a setting,
  wire all four places: dataclass, `to_dict`, `from_dict`, and the `_persist` settings dict
  (+ restore key list) — and the switch/number entity if user-facing. Baseline slots carry
  `schema` (2 = signed `compose_load`); older dumps wipe rather than subtract charge from a
  charge-inflated night median.

- **Head depth is a signed 0.5 K continuum.** Cool device setpoint is
  `internal + head_depth_k` (heat: `internal − depth`). Positive = hysteresis
  residual hold; `0` = SP at live internal; negative = chase
  (`track_delta = −depth`). Under-conditioning steps `+2 → … → 0 → −chase`,
  floored at the **live tracking chase on the 0.5 grid** (not unbounded to
  `−TRACK_DELTA_MAX`). Over-conditioning raises only up to the deepest
  **still COP/residual-useful** depth (`park_residual_max_margin_k` / edge;
  `DEPTH_MAX` while unmapped). If extraction is ~none at the current depth,
  step down only when a shallower depth can restore useful in-band work;
  otherwise off (future overcooling). Coil-wet fan-type and weak residual
  cover step depth down rather than hard soft-release. `pick_depth_k` still
  chooses shallowest covering residual bin when in-band. Shed zones never
  park. Overcorrection release sits `PARK_OVERCOOL_BUFFER_K` below the floor
  (cool).
- **Run-out is the free probe window (when in-band).** A zone wanting off while `min_on`
  forces it to run is parked immediately if already in-band — no sibling requirement, no
  probe budget — because the compressor is alive regardless: observations are free, and an
  idling head saves energy versus tracked run-out. Out-of-band run-out keeps tracking
  (unfinished pull-down must not fan-type park). Parked releases honor `min_on`; continued
  exploitation beyond it requires a live sibling (`want_on`); park direction follows
  `head_mode` when the house's dominant mode goes idle; overcorrected-but-unstoppable zones
  hold at `PARK_MARGIN_MAX_K` rather than being released into a forbidden off. Tracked
  `runout` commands are the `park_learning`-off fallback only.
- **Manual vs HVAC Off.** Hub climate preset `manual` (mirrored switch) stops **all**
  adaptive actuation — comfort commands, parking, fan assist, **and contracted-power
  shedding** — and leaves heads as-is. Thermal/COP/drift estimators still run; park
  learning and park sessions are suspended. Hub HVAC **Off** still force-stops children
  even under Manual (the one exception). Clearing Manual restores the prior preset.
  Shedding while Manual is intentionally off: the user owns the plant.
- **Power is the only honest activity signal.** These heads report `hvac_action: cooling`
  through entire parks while the electrical record is bimodal (fan-only <60 W vs real
  compression >300 W) — the device cycles via an internal hysteresis around its own sensor.
  House `p_ac` is the electrical residual (`p_load − baseline`), never forced to 0 because
  `hvac_action` reports idle. **Charging is the battery branch of `compose_load`**, not a
  known-load entity (listing it there double-subtracts). Discharge is added so AC-on-battery
  stays in `p_load`; charge is subtracted. Shed demand is
  `max(p_grid, 120s peak, max(0, known + residual_p_ac − discharge))` — not
  `max(learned, p_ac)`. Restore uses that same `p_demand`. Night baseline slots learned
  under charge-inclusive `p_load` are wiped on restore (`BaselineModel` schema 2).
  Optional consumption entities are a **Wh ledger** (`EnergyMerge`, max vs sum); 0.25 kWh
  steps are not 60 s watts and never replace `p_ac` / park gating / `StartCounter`.
  Baseline *learning* stays gated on device quiet
  (`_all_off_since`, including fan-only as busy). Park gating / StartCounter prefer a
  *learned* baseline slot (`fallback=False`) so cross-slot medians do not invent phantom AC.
  Parked observations are power-gated (`park.gate_observation`); extraction uses
  `sensible_w` only when freshly fitted after park entry (`sensible_ts`). Never infer
  compression from `hvac_action`. Duty samples run on the 60 s control tick (not the 5 min
  thermal fit), with `PowerDebounce` / `PARK_DUTY_DEBOUNCE_S`. The per-margin `margin_bins`
  map charts hysteresis: `fan_type_min_margin_k()` (shallowest fan-type shelf),
  `residual_max_margin_k()` (deepest still-compressing), `residual_edge_k()` (entry
  target between them), `current_is_fan_type(margin)` (live session class);
  `cop_table_state` buckets house COP by control state **and mode** to audit
  park-hold economics (park learning itself does not read COP tables).
- **Power-react vs 60 s tick.** Meter/known-load changes run `_async_power_react`: head-state
  scan + `_sample_power` + demand finalize + `notify`. Full `controller.tick` runs on that
  path **only when shed engage/release flips** (tracking-delta and park-margin adapt per
  tick and are not `COMMAND_SPACING_S`-gated). The 60 s tick owns `t_rm`, diurnal, baseline
  EWMA, drift, thermal fits, moisture, park learners, and the optional energy Wh ledger
  (`_sample_energy` — never on power-react, never as `p_ac`).
- **Regime policy owns the macro decision.** `_select_regime` picks per tick (15 min dwell):
  `ventilate` when outdoor beats the coolest target by `REGIME_VENT_MARGIN_K` (forces the
  grace-gated window flow — never hard-strands a hot room), `continuous` when aggregate
  `standing_load_w` can feed `REGIME_CONT_LOAD_FRACTION` of `REGIME_FLOOR_PER_HEAD_THERMAL_W
  × total heads` (~160 W/head — West 2-head night loads sit on this edge), else `cycling`.
  `auto_regime` off restores pre-regime semantics exactly. Heat shares the `continuous` path
  when load can feed the floor; `ventilate` remains cool-only. Opt-in `night_ventilate`
  (22:00–08:00) widens the ventilate margin to 0 K and suppresses continuous when outdoor
  is within `NIGHT_SKIP_CONT_K` of the coolest target. Park entry prefers
  `residual_edge_k()` (between residual max and fan-type min) once the map has evidence —
  not the fan-type shelf. `t_out_synthetic` (climatology after a
  configured outdoor dropout) blocks ventilate entry and window suggestions; virgin installs
  without an outdoor entity may still use climatology. Window grace clocks survive brief
  disqualification and re-arm only after `WINDOW_GRACE_S` continuously out of the pool.
- **Latent moisture uses outdoor ACH only.** `outdoor_airflow_m3h` = `k_out × volume` (door/
  exhaust scaled); house-mixing must not couple to outdoor humidity ratio.
- **`cop_by_band` drives COP-timed widen (not precool).** When the active mode's
  outdoor-band COP now differs from a band in the forecast lookahead by
  `COP_ADVANTAGE_ENTER`, the controller sets a hysteretic `cop_widen_k` (capped
  at `COP_WIDEN_MAX_K`) and rebuilds zone bands — **center fixed**. Non-Boost
  widens both edges; **Boost stretches only the conditioning side** (cool: lower
  `lo`; heat: raise `hi`) so the far reactive edge stays Boost-tight — advance
  banks deeper without letting defer float a boosted room warmer/cooler past
  the tight threshold. Timing is `advance` / `defer`; predictive horizons
  stretch/shrink accordingly. Withdraw only after the advantage falls below
  `COP_ADVANTAGE_EXIT` **and** every zone has crossed back inside the unwidened
  band on the widen side (cool-advance below tight `lo` must not snap the floor
  up and look like heat). Mode arbitration runs on the efficiency bands.
  Predictive demand stops at the far edge (cool `temp ≤ lo` / heat `temp ≥ hi`)
  so advance horizons cannot keep digging past the floor — that overshoot was
  what made a later heat flip look like the bug. Cool priors fill gaps; heat
  requires learned `cop_by_band`. Diag: `cop_band_widen_k`, `cop_timing`. Park
  overcool uses the efficiency band. See plan `cop_timed_conditioning_8f3a1c2e`.
- **Zone COP counts latent.** `update_cop` heat flow is sensible + latent; sensible-only
  samples in humid rooms undercount delivered cooling by 30–50% and can fall below
  `COP_MIN`, producing absurd or silently-rejected readings.
- **Control history is want vs action.** Zone `want` is controller desire
  (`off`/`demand`/`helper`); `control_state` is action (`off`/`demand`/`helper`/`park`/
  `runout`/`manual`/…). Action attributes carry `last_reason`, park margins/classification,
  and track depth. Also chart `zone_occupied`, house `effective_preset` / `house_occupied`,
  plus `head_internal_temp`, `head_depth_k`, `track_delta`, `park_margin`,
  `park_extraction`, `park_classification`, `parked_zones`, `operating_regime`,
  `compressor_starts_per_hour`.
- **Mode integrals include on zones.** `demand_integrals` uses no-AC free-float (including
  while conditioning — `predict_free` is the counterfactual) over `MODE_HORIZON_H` (~8 h),
  with temp persistence when confidence is low (for mixed houses). Enter cool/heat above
  `MODE_DEADBAND_KH`; hold until below `MODE_EXIT_KH` — but seasonal demotion is not
  re-promoted by the exit hold. With **no** confident free-float yet, arbitration uses
  `fallback_mode` (outdoor season + indoor deviation), not flat persistence alone — so a
  winter sunlit room cannot command house cooling. Boost beats presence-Away and skips
  vacant band widen; Away/vacant demand setpoints hold near the band edge (not center).
  **House presence** (`presence_adaptation`): vacant house → Away preset; homecoming
  clears track deltas; off disables auto-Away only. **Zone presence**
  (`zone_presence_adaptation`, default on): vacant widen, vacant band-hold, skip
  vacant helpers/fan-assist, and unoccupied weight in mode integrals; off treats
  zone occupancy as unknown (`effective_zone_occupied` → `None`) for those paths
  while sensors still update for diagnostics.
  **Quiet night** (`zone_quiet_night`, per zone, default off): 22:00–08:00,
  heat or cool. This zone is not selected as helper / fan-assist / preferred
  anchor. Out-of-band demand still uses the same comfort band as other rooms.
  In-band standing load may be carried by another zone (vacant cover allowed).
  Two hours before night (`QUIET_NIGHT_BANK_H`), this zone may run to the
  **conditioning hold edge** of that band (cool: `lo + BAND_HOLD_MARGIN_K`;
  heat: `hi − BAND_HOLD_MARGIN_K`) so mix can hold overnight. Boost
  center-seeks and skips quiet-night. Mirrored heads all defer — no office-only
  command.

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
  frame (`park_extraction_w`, residual thresholds); `standing_load_w` is zone-total, so
  exploit compares extraction to `standing_load_w / n_rooms`. Solo-park electrical
  gating uses `fan_floor_w(n_rooms)` (`20 + fan_floor_per_head_w × N`) so multi-head
  fan draw is not mistaken for compression. `StartCounter` uses the same floor.
  Climate entity ids can be swapped vs rooms — coil / EEV / indoor `energy_usage_total`
  follow the physical head, not the HA name. Mixing is observation only
  (`_mixing_free_rider`); a vacant sibling is not a house plant, and indoor kWh
  counters that duplicate outdoor must be merged with `max`, never summed into watts.
- Timezone: runtime uses `local_hour` for diurnal models; timestamps are epoch seconds.
