# Getting started

This guide takes you from a fresh Home Assistant install to Adaptive Comfort
commanding your AC heads. For the feature list see the [README](../README.md);
after you are running, work through [live validation](VALIDATION.md).

**Domain:** `adaptive_comfort` · **Requires:** Home Assistant 2025.4+

## What you need

**Required**

- Existing `climate` entities for each indoor head (the integration drives those;
  it does not replace the vendor integration).
- A whole-house **grid power** sensor in watts (`device_class: power`). This is
  how the controller infers AC draw and enforces contracted-power shedding.
- Floor area of each climatized room (m²). That is the only geometry the model
  needs; UA, capacitance, air exchange, and COP are learned.

**Strongly recommended**

- An **external room temperature** sensor per zone (not the head's internal
  sensor). Heads typically read several kelvin low while cooling.
- An **outdoor temperature** sensor and/or a **weather** entity. Without either,
  outdoor temperature and the short forecast fall back to a generic monthly
  climatology so cold start still works; a real sensor is better.
- Your **contracted supply** in kVA (default 3.45).

**Optional**

- Battery power sensor (discharge is added into house load so AC-on-battery is
  visible; do **not** also list the battery under known loads).
- Known-load power sensors (dishwasher, EVSE, …) subtracted before AC
  disaggregation.
- Indoor AC energy counters (kWh/Wh ledger only — they are not converted to
  60 s watts).
- House presence (`person`, tracker, or binary sensor).
- Per-zone humidity, door, presence, indoor mixing fans, outdoor exhaust fans.

## 1. Install

### HACS (recommended)

1. In HACS, add this repository as a custom repository (category: **Integration**).
2. Search for **Adaptive Comfort** and install.
3. Restart Home Assistant.
4. **Settings → Devices & services → Add integration → Adaptive Comfort**.

### Manual

Copy `custom_components/adaptive_comfort` into `config/custom_components/` and
restart, then add the integration as above.

## 2. House setup

The first form is house-wide. Only grid power is required; everything else
improves the model.

| Field | Typical choice |
|-------|----------------|
| Grid power | Meter-side watts (import). |
| Battery power | Optional. Leave “positive when discharging” on if that matches the sensor. |
| Known loads | Other appliances with a power sensor. Never include the battery here. |
| AC energy / merge | Optional kWh counters. **Max** if each running head copies the outdoor unit's energy; **Sum** if they partition. |
| Outdoor temp / weather | Sensor preferred; weather is the next best forecast source. |
| House presence | Person or occupancy. Used for Away and window suggestions. |
| Multi-split | On when all heads share one outdoor unit (default). |
| Contracted kVA / PF | Your tariff. 3.45 kVA at PF 1.0 is a 3450 W limit. |
| Default target | House comfort center (default 22.5 °C). |

Submit. You get a house device with `climate.adaptive_comfort_house` plus power
and mode diagnostics. **No heads are commanded yet** — add climate zones next.

Reconfigure later from the integration page (⋮ → Reconfigure).

## 3. Add climate zones

On the Adaptive Comfort integration page: **Add climate zone**.

A zone is one or more heads that should be commanded **identically**. Put two
heads in the same zone only when they really share the same room sensor and
should always do the same thing (for example a bedroom and a small office that
share one external sensor). Independent rooms are separate zones.

1. Name the zone.
2. Select the AC `climate` entities (heads).
3. Ceiling height (default 2.6 m).
4. Optional: room temperature (external), humidity, door, presence, fans.
5. **Room areas** — one floor area **per head**. Areas are modeled as separate
   small rooms, never summed. Two 12 m² rooms stay two 12 m² rooms.

No door sensor means the model assumes the door is **open** (inter-room mixing
by default).

Repeat for each climatized area. Each zone gets its own device (corrected
temperature, predicted +60 min, want/action, COP, confidence, …).

## 4. Unconditioned rooms (optional)

**Add unconditioned room** for bathrooms, hallways, and similar — volume and
humidity accounting, not AC control. Mark bathrooms as a **wet room**. An
optional temperature sensor is weak extra evidence for mode selection on a
fresh install.

## 5. First run

1. Open the house climate entity.
2. Set HVAC mode to **Auto** (or Cool / Heat if you want to lock direction).
3. Leave the preset on **None** (or Eco / Away / Boost as you prefer).
4. Add a simple Lovelace card — see [`examples/dashboard.yaml`](../examples/dashboard.yaml).

The hub climate is the thermostat. Do not fight it from the vendor head
entities unless you intend to pause adaptive control.

**Manual** (preset or switch) stops all adaptive commands — comfort, parking,
fan assist, **and** shedding — and leaves heads as you set them. Estimators
keep running. Hub HVAC **Off** still force-stops children even under Manual.

Give it a day. Fit confidence starts low; the first hours use priors and
indoor evidence, then free-float learning when heads are off. Expect:

- All multi-split heads in the **same** mode (never heat + cool together).
- On periods at least the configured min-on (default 20 min).
- `sensor.*_outdoor_effective` with a `source` attribute (`sensor`, `weather`,
  or `climatology`).
- Zone **Action** (`control_state`) showing `demand` / `helper` / `park` /
  `off` rather than the vendor `hvac_action` (those heads often report
  “cooling” while only the fan is running).

Trust **Estimated AC power** (`p_ac`) over `hvac_action` for whether the
compressor is actually working.

## 6. Switches worth knowing on day one

Leave the defaults unless you have a reason. The useful ones:

| Switch | Default intent |
|--------|----------------|
| Multi-split coordination | Same-mode + helpers. |
| Power shedding | Drop heads when the meter approaches contracted kVA. |
| Tracking setpoint control | Dynamic device setpoints (needed on these heads). |
| Park-behavior learning | Learn idle vs residual hold while in-band. |
| Zone enabled (per zone) | Exclude a zone without deleting it. |
| Prefer conditioning other zones at night | Per-zone; skip this room as a helper overnight. |
| Manual control | Pause adaptive actuation. |

Window suggestions, fan assist, night ventilate, and auto regime can wait until
the house is stable.

## 7. If something looks wrong

- **Heads not moving:** house climate is Off or Manual; the zone is disabled; or
  min-off has not elapsed.
- **Wrong season (heat on a hot day):** check Manual was not left on after a
  pulldown; check outdoor source; download diagnostics.
- **MAE / predictor frozen:** scoring only runs while that zone's heads are off
  for a full horizon (15/30/60 min). Conditioning (including Manual) holds the
  last value.
- **Shed too aggressive / too late:** confirm grid sensor is meter-side watts
  and contracted kVA matches the tariff.

**Settings → Devices & services → Adaptive Comfort → ⋮ → Download diagnostics**
and attach the JSON when opening an issue.

## Next

- [Live validation checklist](VALIDATION.md)
- [README](../README.md) — entities, window suggest, fan assist
- [Dashboard examples](../examples/dashboard.yaml)
