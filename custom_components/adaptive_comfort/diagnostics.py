"""Diagnostics dump for debugging model state."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .coordinator import AdaptiveComfortRuntime

REDACT_KEYS = {"latitude", "longitude"}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    runtime: AdaptiveComfortRuntime = hass.data[DOMAIN][entry.entry_id]
    return {
        "entry_data": {k: v for k, v in entry.data.items() if k not in REDACT_KEYS},
        "subentries": {
            sid: {"type": sub.subentry_type, "title": sub.title, "data": dict(sub.data)}
            for sid, sub in entry.subentries.items()
        },
        "runtime": runtime.diagnostics(),
    }
