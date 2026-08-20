---
name: S0 follow-up — vendor-neutral kWh per useful
overview: "Ship battery/shed first. Head energy is a Wh ledger (max vs sum), never 60s p_ac. No η-vs-same-cell watchdog; extend banded helper-COP and existing regime/park/free-ride. Vacant office and East are not house plants — S0 mix into the bedroom failed. The 253 W hold was the bedroom head, not the office."
todos:
  - id: battery-accounting
    content: "Signed compose_load (charge subtract once). Battery entity is not a known load. p_demand third term = known+p_ac−discharge; stop max(learned, p_ac) for that slot. Restore on p_demand. Baseline schema wipe/decay."
    status: completed
  - id: head-energy-merge
    content: "Optional consumption entities: house Wh. Merge max (duplicate outdoor) vs sum (partitioned). Never replace 60s p_ac from 0.25 kWh steps."
    status: pending
  - id: helper-cop-banded
    content: "Extend consolidated_by_cop_table to banded / kWh per K·h (cross-N). No new N2 latch. No copy of _select_regime. Keep _mixing_free_rider; do not start vacant East/office so West can ride mix."
    status: pending
  - id: reject-blunt-n1
    content: "Do not coast solely because p is 180–320 W. 253 W bedroom hold was useful."
    status: completed
  - id: tests-agents
    content: "Charge/discharge shed; no double-subtract; dirty baseline; AGENTS.md. (0.25 kWh ≠ 15 kW p_ac waits on cut 2)"
    status: completed
isProject: false
---

# S0 follow-up — vendor-neutral kWh per useful heat

Evidence: [docs/s0-reconciliation-night.md](../../docs/s0-reconciliation-night.md),
history(16) battery series, dossier N1–N6.

Independent review ([plan review](ffee8f42-6a39-465e-b058-c2e972df9912)): **rework.**
Only cut 1 ships in this shape. Quantized head kWh must not replace 60 s
`p_ac`. Do not add a delivered-per-watt latch against the same COP cell
that was trained on the wasteful 3-head floor. One review line is stale:
the 253 W hold was the **bedroom** head (22:49), not the office.

## Constraint (non-negotiable)

The tick must run without vendor debug (coil, EEV, Hz, operating current).
Those may exist for offline scoring; they are not `controller.tick` inputs.
No swap maps, no `cop_table_hz` as a control key, no
`q̂ ∝ (room − coil) × fan` allocator.

Optional energy entities are allowed (same class as known loads). They are
not Mitsubishi-specific. **How they merge is install-specific** (see cut 2).

## Goal

Raise kWh of useful heat/cool per kWh electrical. Extend existing
mechanisms; each cut names the ledger that will kill or keep it.

## West identity (do not trust HA names)

Office started **03:34** (`climate.bedroom` on, HA office EEV/coil open).
**22:49–03:25 is the bedroom head** (`climate.office` on, HA bedroom
EEV/coil open). Climate entity ids are swapped vs the rooms; coil / EEV /
`energy_usage_total` follow the physical unit on this night. Earlier notes
that treated climate as canonical and remapped refrigerant inverted the two
West runs.

## Mix (regular night, not SP 18)

S0 forced SP 18 so heads would run. That chase is not the consolidation
experiment. On a normal night the question is one well-running head vs
West cycling. S0 still measures **coupling**:

| Feeder | Bedroom air (UltimateSensor) |
|---|---|
| East-only 21:18–22:47 | −0.19 K/h (East 2.8 K colder). Weak. |
| Bedroom-only 22:49–03:25 | 24.3 → 20.7, then hold ~20.7 °C at ~253 W. Local evaporator. |
| Office-only 03:34–08:07 | 21.2 → 24.1. Office internal ~21 °C. Mix **failed**. |

Do **not** start vacant East or the empty office at a deeper-than-comfort
SP so the bedroom can “ride mix.” `_mixing_free_rider` already drops a
zone when a sibling covers hold; S0 says that cover is false for East →
West and office → bedroom. Demand follows the out-of-band occupied zone
(bedroom). Opening West still commands both mirrored heads; this plan does
not add an office-only production path.

## What S0 actually taught (control-relevant)

| Observation | Reading | Do **not** conclude |
|---|---|---|
| 3-head floor wasted work (dossier) | Extra heads can be overhead | Must read EEV/Hz; detect it by comparing this cell to *itself* |
| ~253 W hold, bedroom sensor fell | Bedroom head in the sensed room; low watts can be useful | `p` ∈ [180, 320] is always dead |
| Office-only 03:34, bedroom sensor rose | Office does not cool the bedroom; internal 21 °C is not the room | Empty office at a deep SP is a house plant |
| Living 680 W, East pulldown | Hard one-head chase is valid | Always one head |
| Charge +1035 W → residual `p_ac` 1132 W | Residual includes charge | Need outdoor current |
| Discharge, import 300–500 W | Contract is import | Count discharge in `p_demand` |
| Indoor kWh **duplicates** outdoor per running head; sum ≈ 1.73× | `max(delta)` for *this* lie | `max(delta)` for every vendor (partitioned kWh would under-count) |

Cooktop: out of scope. Known-loads already exist; it was not wired.
S0 office-only is the mix test (failed). Production still never commands a single West head.

## Ranked cuts

### 1. Battery vs limiter — implement first

Battery entity is already `sensor.main_home_battery_battery_power` with
`battery_positive_discharging: false`. **It is not a known-load entity.**

Change `compose_load` to a **signed** battery term (drop
`load += max(0, discharge)`). This house: `load = p_grid − p_battery − known`
because positive `p_battery` is charge and negative is discharge (so
`− p_battery` subtracts charge and adds discharge). Equivalent form:

```
discharge = p_batt if positive_discharging else -p_batt
load = p_grid + discharge - known   # discharge may be negative (= charge)
load = max(0, load)
```

Do not list the battery sensor in `known_load_entities`. That double-subtracts
charge and, on discharge, can *add* it twice if the reading is negative.

**Baseline.** `BaselineModel.update` learned night slots on charge-inclusive
`p_load`. After the sign fix, `p_ac = clean_load − dirty_baseline` → 0 until
EWMA (`alpha=0.1`) forgets. Wipe or decay those slots (same idea as
`moisture_schema`). Required in this cut, not a later N6 hold.

**Shed.** Today `_finalize_demand` uses
`ac_w = estimated_active_ac_draw_w(draws, p_ac)` then
`contracted_demand_w(..., known, ac_w)` i.e. `max(p_grid, peak, known + max(learned, p_ac))`.
The learned-draw max is why battery evenings can still shed on ~1 kW house
fiction while import is 300 W.

Replace the third term only:

`p_demand = max(p_grid, p_grid_peak_120s, max(0, known_sum + p_ac − discharge))`

`p_ac` here is the **residual** (after signed `compose_load`), not quantized
head energy, not `max(learned, p_ac)`.

**Restore.** `restore_allowed` is already passed `p_demand` (param is misnamed
`p_grid`). After the third term is honest, keep restore on that `p_demand`.
Rename the parameter so the next author does not “fix” it to raw import by
accident. Add a test: large learned draws, `p_grid` 200 W, restore allowed.

Adjudicated by: overnight charge, residual `p_ac` does not track `p_battery`;
discharge, no shed while `p_grid` is under 92%/96% of 3450 W.

Update `test_compose_load_with_battery_discharge`: `compose_load(1200, -500, True) == 1200`
**is the bug**.

### 2. Head energy — Wh ledger, not 60 s watts

S0 counters step **0.25 kWh**. At ~250 W that is one click per hour.
`W = ΔWh/Δt` on a 60 s tick is 0 then ~15 kW. That must not feed park
gating, `StartCounter`, shed, or any watchdog. Power-react also would not
see kWh clicks.

- Optional consumption entity list. Expose house **Wh**.
- **Two merge modes:** `max` when each head duplicates outdoor (this WF-RAC);
  `sum` when heads partition real energy. Configure; do not ship `max` as
  the only law.
- Ignore NaN/negative; reset downward → new origin, add 0; else add
  `max` or `sum` of positive deltas per the mode.
- Residual `p_ac` stays the 60 s electrical truth after cut 1.

Adjudicated by: 1-head night, merged Wh ≈ residual kWh; 2-head, merged Wh ≈
residual (max mode) not 1.7×.

### 3. Extra heads — extend helper COP, not a new latch

Do **not** add house `η` vs the median of the current `cop_table_*` cell.
Those cells *are* the 3-head floor. The watchdog would match the waste.
`HOUSE_COP_MIN_POWER_W = 250` also means the cheap hold barely files, so
the median is the high-watt mass.

Do **not** re-check `standing_load < 0.6 × 265 × N_open` as a second gate.
That inequality already *is* `_select_regime` (continuous vs cycling).
Park residual / `_park_exploit_ok` / `_adapt_head_depth_k` already drop
idle and weak cover.

**Extend** `consolidated_by_cop_table` (`controller.py` ~1466): today it
drops helpers when unbanded `cop_by_head_count` says demand-only beats
demand+helpers. Make that comparison **banded** (or kWh/K·h) — *cross-N*,
the only comparison that can see extra heads as overhead. Evidence gate:
house kWh/K·h and band-breach minutes, not unbanded `cool|N`.

`q̂` is already `c_eff · (dT/dt − free_float)` in `sensible_power_w`, and
West is multiplied by `n_rooms`. Do not add a parallel “West dT must fall”
predicate.

Reject N1 as a trigger (180–320 W dead zone). Depth + park residual stay
the low-load modulators. The 253 W hold that must survive is **bedroom
local**, not mix from East or office.

Do not add a “consolidate on vacant East/office” actuator. Free-ride stays
`_mixing_free_rider` (mix must actually cover). Occupied bedroom out of
band is demand.

### 4. Hold

N3 edge-ride, N4 RH bursts, N5 MAE, night ventilation, tracking/park
(morning 19.5 °C was Manual SP 18), vacant-zone-as-plant, office-only
production command. All-off relearn is **not** held — it is cut 1.

## Optional vendor extras (never in `tick`)

Coil/EEV/Hz → diagnostics only. This install: `climate.office` is the
bedroom head; `climate.bedroom` is the office. Footnote, not a tick map.

## Tests (core only)

- Charge +800 W, AC 250 W, battery **not** in known loads → `p_ac` ≈ 250;
  `p_demand` follows `p_grid` (~1050).
- Battery also in known loads must not double-subtract (guard test).
- Discharge −800 W, AC 250 W, `p_grid` 200, known 100, **large learned
  draws** → third term 0, `p_demand` ≈ 200; restore allowed.
- Dirty baseline 900 W, post-fix `p_load` 300 → slot wiped/decayed, not
  `p_ac` ≈ 0 forever.
- Two heads both +0.25 kWh, max mode → house Wh +0.25, **not** a 15 kW
  `p_ac` sample.
- Negative / reset-to-zero add nothing.
- Low watts + on-ledger η (existing sensible) → no new watchdog (there
  isn’t one).
- No office-only fixture, no vacant-East-as-plant fixture, no cooktop gate.

## AGENTS.md (when implementing)

- `core/` stays vendor-neutral. Head internals are diagnostic, not control.
- Charging is the **battery branch** of `compose_load`, not a known-load
  entity. Shed:
  `max(p_grid, peak, max(0, known + residual_p_ac − discharge))`.
  Restore on that `p_demand`. Night baseline slots must relearn after the
  sign change.
- Indoor kWh is never summed on this plant; merge mode is configured.
  0.25 kWh is not 60 s W.
- Extra-head overhead is cross-N helper COP / regime / park residual, not
  a watt band and not η vs the same COP cell.
- This house: climate ids are swapped vs West rooms. Office → bedroom mix
  failed on S0; do not treat vacant East/office as a house plant.
- `_mixing_free_rider` is the mix path; do not add a parallel consolidate-
  on-empty-room policy.
