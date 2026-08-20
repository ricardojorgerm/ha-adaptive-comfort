# Adaptive Comfort

Self-learning whole-home thermostat for Home Assistant. Coordinates multiple AC heads (especially multi-split systems), learns each room's thermal dynamics from sensor data alone, and optimizes for comfort and efficiency.

**Domain:** `adaptive_comfort` · **Requires:** Home Assistant 2025.4+

**New install?** Follow [Getting started](docs/GETTING_STARTED.md).

## Features

- **One house climate entity** — set target, mode (off/heat/cool/auto), and presets (eco/away/boost/manual); the integration drives your existing AC `climate` entities.
- **Minimal configuration** — only room floor areas are required model inputs; everything else (UA, capacitance, air exchange, COP, latent load) is inferred.
- **Multi-split aware** — all heads share one HVAC mode; optional fan assist for opposite-demand zones (respects wet-coil latent penalty).
- **Model-driven mode selection** — predicts free-float drift instead of outdoor-temperature hysteresis; works from cold start with sensible priors and a climatology fallback.
- **Power shedding** — sheds AC load against contracted kVA when grid power exceeds your limit.
- **Optional open-window alerts** — when outdoors is much colder than indoors and someone is home, suggests opening windows instead of cooling (notification-ready binary sensor).
- **Rich diagnostics** — k, ACH, UA, COP, sensible/latent power, drift offsets, model confidence per zone.

## Installation

See [Getting started](docs/GETTING_STARTED.md) for a full walkthrough. Short version:

### HACS (recommended)

1. Add this repository as a [custom repository](https://hacs.xyz/docs/faq/custom_repositories/) in HACS (category: **Integration**).
2. Search for **Adaptive Comfort** and install.
3. Restart Home Assistant.
4. Go to **Settings → Devices & services → Add integration** and search for **Adaptive Comfort**.

### Manual

Copy `custom_components/adaptive_comfort` into your Home Assistant `config/custom_components/` directory and restart.

## Configuration

### 1. House (main entry)

| Setting | Required | Notes |
|---------|----------|-------|
| Grid power sensor | Yes | Whole-house meter reading in W (used for AC inference and shedding) |
| Battery power sensor | No | Optional; discharge adds to available load for AC estimation |
| Known load sensors | No | Subtract monitored appliances before disaggregating AC power |
| Outdoor temp / weather | No | Improves model; climatology fallback when missing |
| House presence | No | Enables away/eco presets and open-window suggestions |
| Multi-split toggle | Yes (default on) | Enables same-mode coordination and fan assist |
| Contracted kVA | Yes (default 3.45) | 3.45 kVA × PF 1.0 → 3450 W limit |
| Default target | Yes (default 22.5 °C) | House comfort center |

### 2. Climate zones (subentries)

Add one zone per climatized area. Each zone can include **one or more AC heads** that mirror each other.

For each head you enter the **room area in m²** separately — areas are **not summed**. A zone with two 12 m² rooms is modeled as two small rooms, not one 24 m² room.

Optional per zone: external temperature sensor, humidity, door, presence, indoor circulation fans, outdoor exhaust fans.

### 3. Unconditioned rooms (subentries)

Add bathrooms, hallways, etc. for whole-house volume and humidity accounting. An optional temperature sensor provides weak auxiliary evidence for mode selection on cold start (weight 0.3 vs 1.0 for conditioned zones).

### Reference setup (author)

- 3-head multi-split (one outdoor unit)
- Grid power sensor + optional battery
- External temp sensors on two conditioned rooms
- Bathroom temperature sensor on an unconditioned room
- Contracted power 3.45 kVA

## Entities

### House device

| Entity | Purpose |
|--------|---------|
| `climate.*` | House-wide thermostat (target, mode, preset) |
| `sensor.*_house_load` | Total house load (W) |
| `sensor.*_ac_power_estimate` | Estimated AC draw |
| `sensor.*_power_headroom` | Headroom below contracted limit |
| `sensor.*_outdoor_effective` | Outdoor temp (attribute: source = sensor/weather/climatology) |
| `sensor.*_dominant_mode` | Current arbitration mode (off/heat/cool) |
| `binary_sensor.*_window_suggestion` | Open windows suggested (with zone names attribute) |
| `binary_sensor.*_shedding_active` | Load shedding in progress |
| Switches | Coordination, presence adaptation, shedding, tracking, park learning, auto regime, night ventilate, fan assist, window suggest, manual control |
| Numbers | Comfort band, adaptive blend, min on/off, contracted kVA, shed thresholds, fan floor per head |

### Zone devices

Corrected temperature, predicted +60 min, k, ACH, UA, COP, sensible/latent power, drift offset, fit confidence, want/control_state (with park/track attributes), head internal temp, park margin/extraction/classification, conditioning/shed/window-suggestion/occupied binary sensors, comfort offset, zone enabled.

### Regime and parking (summary)

With **auto regime** on, the controller picks `ventilate` / `continuous` / `cycling` from outdoor vs target and standing load. Parking holds a satisfied multi-split head above its internal sensor to learn idle vs residual behavior (electrically gated); see AGENTS.md for the full semantics.

### Manual preset

**Manual** (climate preset or switch) stops all adaptive actuation — comfort commands, parking, fan assist, and contracted-power shedding — and leaves heads as you set them. Estimators keep learning in the background. Hub HVAC **Off** still force-stops children even under Manual.

## Optional features

### Suggest opening windows

Enable the **Suggest opening windows** switch on the house device. When dominant mode is cool, a room is ≥ 2 K warmer than outdoors, and presence is detectable, the integration:

1. Turns on `binary_sensor.*_window_suggestion` (use in automations/notifications).
2. Holds mechanical cooling for that zone for 15 minutes.
3. Cools anyway after the grace period if the window stays closed.

If the switch is off, or no presence entity is configured, the integration always cools and never assumes you can open a window.

### Fan assist (multi-split)

When one zone needs cooling and another is too cold, the cold zone may receive `fan_only` to mix house air — but only after the coil has been dry for 30 minutes (avoids re-evaporating condensate). Disable via the **Fan assist** switch.

## Dashboard

See [`examples/dashboard.yaml`](examples/dashboard.yaml) for Lovelace cards and a sample automation that notifies when opening windows is suggested.

Quick start — add a **Entities** card:

```yaml
type: entities
title: Adaptive Comfort
entities:
  - entity: climate.adaptive_comfort_house
  - entity: sensor.adaptive_comfort_house_load
  - entity: sensor.adaptive_comfort_power_headroom
  - entity: sensor.adaptive_comfort_dominant_mode
  - entity: sensor.adaptive_comfort_house_cop
  - entity: binary_sensor.adaptive_comfort_window_suggestion
  - entity: binary_sensor.adaptive_comfort_shedding_active
```

Replace entity IDs with yours (Settings → Devices → Adaptive Comfort → copy entity IDs).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install pytest ruff
pytest tests -q                    # core logic (no HA required)
pip install -r requirements_test.txt
pytest tests -q                    # includes config-flow tests (needs HA test harness)
ruff check . && ruff format .
```

CI runs ruff, hassfest, HACS validation, and pytest on Python 3.13.

## Diagnostics

**Settings → Devices & services → Adaptive Comfort → ⋮ → Download diagnostics** exports model state, subentry config, and runtime snapshot for debugging.

## Live validation

After installing via HACS, follow [Getting started](docs/GETTING_STARTED.md), then the checklist in [`docs/VALIDATION.md`](docs/VALIDATION.md).

## License

MIT
