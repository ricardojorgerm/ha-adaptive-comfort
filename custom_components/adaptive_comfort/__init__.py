"""Adaptive Comfort: self-learning whole-home AC thermostat."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, PLATFORMS
from .coordinator import AdaptiveComfortRuntime


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    runtime = AdaptiveComfortRuntime(hass, entry)
    await runtime.async_setup()
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload when the entry or its subentries change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        runtime: AdaptiveComfortRuntime = hass.data[DOMAIN].pop(entry.entry_id)
        await runtime.async_unload()
    return unload_ok
