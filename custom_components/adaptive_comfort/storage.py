"""Persistence of learned model state via the HA Store helper."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN, STORAGE_VERSION


class AdaptiveComfortStore:
    """Versioned storage blob, one per config entry."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}")
        self.data: dict[str, Any] = {}

    async def async_load(self) -> dict[str, Any]:
        self.data = await self._store.async_load() or {}
        return self.data

    async def async_save(self, data: dict[str, Any]) -> None:
        self.data = data
        await self._store.async_save(data)

    def async_delay_save(self, data: dict[str, Any], delay_s: float = 60.0) -> None:
        self.data = data
        self._store.async_delay_save(lambda: self.data, delay_s)

    async def async_remove(self) -> None:
        await self._store.async_remove()
