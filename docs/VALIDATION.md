# Live validation checklist

Use this after installing via HACS or manual copy on your Home Assistant instance.

## Install

- [ ] Integration appears in HACS (or manual copy under `custom_components/adaptive_comfort`)
- [ ] Restart completes without errors in the log
- [ ] **Settings → Devices & services → Add integration → Adaptive Comfort** opens the house setup form

## House setup

- [ ] Grid power sensor selected; entry creates successfully
- [ ] House device appears with climate entity and diagnostic sensors
- [ ] `sensor.*_outdoor_effective` shows a value (sensor, weather, or `climatology` source attribute)

## Zones

- [ ] Add zone subentry with 1–3 climate heads
- [ ] Per-head room areas step appears (not a single summed area)
- [ ] Zone device created; entities linked to zone subentry device
- [ ] AC heads receive commands when house climate mode is set to cool/heat/auto

## Unconditioned room

- [ ] Add bathroom (wet room) with optional temp sensor
- [ ] Aux sensor contributes to mode selection (check `sensor.*_dominant_mode` attribute `source` includes fallback when models are cold)

## Multi-split behaviour

- [ ] All heads switch to the same mode (never heat + cool simultaneously)
- [ ] Fan assist: when one room is cold during house cooling, fan-only may run on that head (after coil dry period)
- [ ] **Fan assist** switch disables fan-only commands

## Open windows (optional)

- [ ] Enable **Suggest opening windows** switch
- [ ] With presence on and indoor > outdoor + 2 K: `binary_sensor.*_window_suggestion` turns on
- [ ] Cooling for that zone pauses ~15 min, then resumes
- [ ] Automation from `examples/dashboard.yaml` fires notification (if configured)

## Power shedding

- [ ] Simulate or wait for grid power above 92% of contracted limit
- [ ] `binary_sensor.*_shedding_active` turns on; zones shed in priority order
- [ ] AC restores when power drops below restore threshold

## Diagnostics

- [ ] Download diagnostics from integration page; JSON includes `runtime`, `subentries`, model state

## Cold start (first 24 h)

- [ ] Rooms move toward comfort band without manual intervention
- [ ] No rapid head cycling (on periods ≥ configured min-on, default 20 min)
- [ ] Dominant mode reasonable for season (cool on hot day, heat on cold night)

## Log watch

```text
# No recurring errors expected; filter:
logger: custom_components.adaptive_comfort
```

Report issues at https://github.com/ricmacas/ha-adaptive-comfort/issues with diagnostics attached.
