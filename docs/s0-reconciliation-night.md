# S0 reconciliation night — 19–20 Aug 2026

Manual sequential single-head runs on the Mitsubishi Heavy SCM41ZS-W (3 heads).
Planned as the dossier’s reconciliation hour (30 min all-off, then 1 h per head).
Ran longer because the house was left in that state overnight.

**Window.** 19 Aug 20:00 → 20 Aug 09:58 Europe/Lisbon.
**Sources.** `history(15).csv` (1-min resample), adaptive-comfort config dump `(21)`,
WF-RAC config dumps, project history, `scm41-head-internals-dossier-2`.
**Controller.** Preset `manual` from 20:35. Estimators still ran; no adaptive actuation.

Overnight traces are also in the Cursor canvas `s0-reconciliation-night` (not in this repo).

---

## Standing facts for this export

1. **West climate names are swapped vs the rooms.** Lived sequence: office
   started **03:34**. That is `climate.bedroom` (internal ~21 °C, HA office
   EEV/coil open). The 22:49 run is the **bedroom** head: `climate.office`
   on, HA bedroom EEV/coil open, bedroom UltimateSensor 24.3 → 20.7. Coil,
   EEV, and `energy_usage_total` follow the physical room on this night;
   climate entity ids do not. Earlier notes that treated climate as canonical
   and remapped refrigerant inverted the two West runs.
2. **Cooktop 22:00–22:20** during the living-only window is not in known-loads,
   so it landed in `p_ac`. Those 21 minutes are dropped from every electrical
   total.
3. **Main-battery charging is not subtracted from `p_load`.** The integration
   uses `sensor.main_home_battery_battery_power` with
   `battery_positive_discharging: false`. `compose_load` only *adds* discharge;
   charging stays in `p_grid` and therefore in `p_ac`. That export did not
   include the main battery entity — only EcoFlow `load_from_battery`
   (discharge, 25% coverage). **23:30–00:14** is the charging signature:
   `p_ac` jumped ~880 W while outdoor current stayed 1.10 A (~253 W) and Hz
   stayed 23. Extra in `p_ac` vs I×230 V: **0.66 kWh**. Discharge windows
   (living ~21:20–22:30, morning 08:00–09:50, ~300–450 W from the EcoFlow
   sensor) were composed correctly: `p_ac` matched I×230 V there. Other large
   grid-minus-load gaps (~1.0–1.7 kW at 22:30–23:10, 00:40, 02:50, 05:30)
   look like a *metered* known load (possibly portable-battery AC input) and
   did not contaminate `p_ac`.
4. **Device counters reset to 0** when the heads were switched off at 20:36.
   Deltas below are from that origin. Quantization is 0.25 kWh.
5. **Operating current is shared.** `i_bedroom`, `i_office`, and `i_living`
   agree (median |Δ| = 0). `I × 230 V` is the outdoor electrical referee.
   `freq = 0` is authoritative off; EEV parks at ~78 while stopped, so
   `EEV > 0` is not “feeding.”

---

## Protocol as run

| Lisbon | Min | What ran | Hz | Outdoor W (I×230) | Room |
|---|---:|---|---:|---:|---|
| 20:00–20:35 | 36 | West still on (pre-manual) | 20 | ~195 | West 23.5 → 22.8 |
| 20:37–21:17 | 41 | All off (baseline) | 0 | 0 | West +2.6 K/h free-float |
| 21:18–22:47 | 90 | Living only, SP 18 | 55 | 631 | East 25.1 → 21.4 |
| 22:49–03:25 | 277 | Bedroom head only | 23 | 253 | West 24.3 → 20.7, then hold |
| 03:34–08:07 | 274 | Office head only | 29 | 379 | West 21.2 → 24.1, failed hold |
| 08:08–09:52 | 105 | Both West heads | 50 | 568 | West 24.1 → 19.5 |

All-off baseline: house load median 46 W, `p_ac` median 11 W, Hz = 0, I = 0.
That is a usable night slot for N6 (scheduled re-anchor), even though it was
41 min rather than 30.

---

## Verdicts against the Aug 17–19 dossier

### C1 — device vs disaggregator

**Not “the disaggregator is 34% low.”** Each indoor `energy_usage_total`
tracks outdoor energy while that head is on. When two heads run, both
registers increment at that same outdoor rate, so the **sum is almost 2 ×
outdoor**. The 17.0 vs 12.73 kWh gap in the 44 h window is this overlap
during 3-head operation, not a broken residual.

| Run | Device kWh | I × 230 V | p_ac | Device / I×V |
|---|---:|---:|---:|---:|
| Living (90 min; cooktop dropped from p_ac) | 0.75 | 0.95 | 0.77 | 0.79 |
| Bedroom head (4.6 h) | 1.00 | 1.18 | 1.02 clean / 1.87 raw | 0.85 |
| Office head (4.6 h) | 1.50 | 1.72 | 1.69 | 0.87 |
| Both West heads (1.8 h) | **1.75 sum** | 1.01 | 1.02 | **1.73** |

One running head is a usable outdoor proxy after a ~0.85 calibration against
current (power factor plus 0.25 kWh quant). **Do not sum indoor counters
across heads.**

### C2 — East vs West money

**No East/West inversion in this isolated test.** Zone allocation followed
the climate that was actually on (living → East, West heads → West). The
historical “devices West-heavy, integration East-heavy” split is an
allocator-weight problem under simultaneous 3-head running, not swapped
East/West wiring. Within West, climate entity ids are swapped vs rooms;
`energy_usage_total` on this night follows the physical head.

### H1 — ultra-part-load recirculation

**Confirmed in the inverse.** The pinched-valve, ~19 °C coil, 20–40 Hz
regime is a **3-head floor effect**. One open head takes the minimum
compressor flow:

| Sole feeder | Hz | EEV | Coil °C | Hall dewpoint | Outdoor W |
|---|---:|---:|---:|---:|---:|
| Living | 55 | 90 | 6.4 | 16.8 | 631 |
| Bedroom head | 23 | 44 | 7.8 | 16.8 | 253 |
| Office head | 29 | 46 | 7.3 | 16.4 | 379 |

Bedroom-head stays on the frequency floor *and* still opens the valve and
drops the coil below dewpoint. Same minimum hertz as the dossier’s dead
zone — different valve. Head count, not frequency, is the modulation lever
below ~30% of nameplate.

Living-only at SP 18 vs internal 28 left the floor entirely (55 Hz). A
static deep setpoint still overshoots once the room is in band; tracking /
park remains the way to command a small error.

### H2 — West latent monopoly

**A 3-head pinched-valve result, not a hardware limit.** West coils went to
7–8 °C as soon as that head was the sole feeder — 8–10 K below hallway
dewpoint. Living remains the colder coil when it is the one open (6.4 °C,
EEV 90).

---

## Per-run notes

**Living only.** Huge commanded error. Valve to 90, coil 6.4 °C, compressor
off the floor. East −3.6 K in 69 clean minutes. Air-node COP ~0.8 from
east `c_eff` = 133 Wh/K (storage + mix, no diurnal, no latent). A 6 °C
coil against 16.8 °C dewpoint is doing latent work the air node does not
see.

**Bedroom head only (22:49).** The head in the sensed room. Pulldown 24.3 →
20.7 in ~25 min at 23 Hz / 253 W, then a 4 h hold around 20.7 °C. Bedroom
internal (HA `climate.office`) 27.5 → ~22. Idle office internal stayed
~27 °C (warm cassette, not a room witness). This is the consolidation
posture the dossier asked for — local evaporator, not mix from the office.

**Office head only (03:34).** Coil 7.3 °C, EEV 46, 379 W — office internal
(HA `climate.bedroom`) satisfied (~21 °C) and the bedroom air still warmed
21.2 → 24.1. Free-float from the evening all-off was +2.6 K/h; this cut it
to about +0.6 K/h, not a hold. Office → bedroom mix failed. Do not treat
the office internal 21 °C as bedroom comfort. West `c_eff` at 30 Wh/K in
the live dump is too small against the July moisture-anchored C; west
thermal watts tonight are a lower bound.

**Both West heads, morning.** SP 18, 50 Hz, both valves open (31 / 42),
both coils 6.9 °C, West 24.1 → 19.5 °C. The always-on overshoot from a
manual deep setpoint rather than the controller. Device sum 1.75 kWh vs
outdoor 1.01 kWh — the C1 overlap in one clean interval.

---

## What S0 does not settle

- No matched 1 h × 3 isolated runs, so per-head COP at equal outdoor
  conditions is not adjudicated.
- West `c_eff` (30 Wh/K) vs July C (~70 Wh/K per 9 m² room) is unresolved.
- Cooktop still has no power sensor in known-loads.
- `compose_load` does not subtract main-battery charging; 23:30–00:14 (~0.66 kWh
  in `p_ac`) is that hole. The main battery power entity was not in this export.
- Refrigerant telemetry is on the recorder, not yet in the controller.

---

## Lineage

Thermometers lie (drift) → `hvac_action` lies (park gating) → `active_ratio`
lies (hysteresis map) → “running” lies (3-head recirculation) → **indoor
kWh counters lie when summed** (this night). Outdoor current and `freq = 0`
are the witnesses that survived. The energy meter remains the only
electrical instrument never impeached; I × 230 V is its 1-minute proxy.
