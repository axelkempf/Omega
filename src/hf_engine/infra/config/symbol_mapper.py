from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, MutableMapping, Optional, Tuple


class SymbolMappingError(ValueError):
    """Allgemeiner Fehler für Mapping-Probleme."""


@dataclass(frozen=True)
class _Config:
    """Konfigurationsschalter für das Verhalten bei Missing-Keys."""

    # Wenn True: fehlende Keys führen zu KeyError, andernfalls Passthrough (eingehender Wert wird zurückgegeben)
    strict_missing: bool = False


def _require_str_dict(name: str, m: Mapping) -> None:
    """Validiert, dass Mapping-Keys und -Values Strings sind."""
    if not isinstance(m, Mapping):
        raise TypeError(f"{name} muss ein Mapping sein, erhalten: {type(m).__name__}")
    for k, v in m.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError(
                f"{name} muss str->str abbilden, erhalten: {type(k).__name__}->{type(v).__name__} für {k!r}:{v!r}"
            )


def _build_inverse(name: str, m: Mapping[str, str]) -> Mapping[str, str]:
    """Erzeugt das inverse Mapping mit Kollisionserkennung."""
    inverse: dict[str, str] = {}
    collisions: dict[str, Tuple[str, str]] = {}  # value -> (first_key, second_key)
    for k, v in m.items():
        prev = inverse.get(v)
        if prev is not None and prev != k:
            collisions[v] = (prev, k)
        inverse[v] = k
    if collisions:
        details = ", ".join(
            f"{val!r} durch {a!r} und {b!r}" for val, (a, b) in collisions.items()
        )
        raise SymbolMappingError(
            f"Inverses Mapping für {name} ist nicht eindeutig (Werte-Kollisionen): {details}"
        )
    return MappingProxyType(inverse)


class SymbolMapper:
    """
    Robuster Mapper zwischen logischen Symbolen und Broker-/Datafeed-Symbolen.

    - Immutabel: Interne Maps sind read-only (MappingProxyType).
    - Validierung: Prüft str->str, erkennt Kollisionen beim Invers-Mapping.
    - Fehlende Einträge:
        * strict_missing=False (Default): Passthrough (Eingabe unverändert).
        * strict_missing=True: KeyError.
    """

    __slots__ = ("_broker_map", "_data_map", "_broker_inverse", "_data_inverse", "_cfg")

    def __init__(
        self,
        broker_map: Mapping[str, str],
        data_map: Optional[Mapping[str, str]] = None,
        *,
        strict_missing: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        broker_map:
            Mapping logisches Symbol -> Broker-Symbol.
        data_map:
            Mapping logisches Symbol -> Datafeed-Symbol. Fällt auf broker_map zurück, wenn None.
        strict_missing:
            Wenn True, werden nicht bekannte Symbole mit KeyError quittiert; sonst Passthrough.
        """
        _require_str_dict("broker_map", broker_map)
        if data_map is not None:
            _require_str_dict("data_map", data_map)

        # Read-only Proxies
        broker_map_ro = MappingProxyType(dict(broker_map))
        data_src = broker_map if data_map is None else data_map
        data_map_ro = MappingProxyType(dict(data_src))

        broker_inv_ro = _build_inverse("broker_map", broker_map_ro)
        data_inv_ro = _build_inverse("data_map", data_map_ro)

        object.__setattr__(self, "_broker_map", broker_map_ro)
        object.__setattr__(self, "_data_map", data_map_ro)
        object.__setattr__(self, "_broker_inverse", broker_inv_ro)
        object.__setattr__(self, "_data_inverse", data_inv_ro)
        object.__setattr__(self, "_cfg", _Config(strict_missing=strict_missing))

    # --- Öffentliche Properties (read-only) ---
    @property
    def broker_map(self) -> Mapping[str, str]:
        """logisch -> broker (read-only)"""
        return self._broker_map

    @property
    def data_map(self) -> Mapping[str, str]:
        """logisch -> datafeed (read-only)"""
        return self._data_map

    @property
    def broker_inverse(self) -> Mapping[str, str]:
        """broker -> logisch (read-only)"""
        return self._broker_inverse

    @property
    def data_inverse(self) -> Mapping[str, str]:
        """datafeed -> logisch (read-only)"""
        return self._data_inverse

    # --- Mapping-Funktionen ---
    def to_broker(self, logical: str) -> str:
        """Übersetzt logisches Symbol -> Broker-Symbol."""
        return self._lookup(self._broker_map, logical)

    def to_datafeed(self, logical: str) -> str:
        """Übersetzt logisches Symbol -> Datafeed-Symbol."""
        return self._lookup(self._data_map, logical)

    def to_logical_from_broker(self, symbol: str) -> str:
        """Übersetzt Broker-Symbol -> logisches Symbol."""
        return self._lookup(self._broker_inverse, symbol)

    def to_logical_from_datafeed(self, symbol: str) -> str:
        """Übersetzt Datafeed-Symbol -> logisches Symbol."""
        return self._lookup(self._data_inverse, symbol)

    # --- Interne Helfer ---
    def _lookup(self, m: Mapping[str, str], key: str) -> str:
        if not isinstance(key, str):
            raise TypeError(f"Symbol muss str sein, erhalten: {type(key).__name__}")
        value = m.get(key)
        if value is None:
            if self._cfg.strict_missing:
                raise KeyError(f"Symbol nicht im Mapping gefunden: {key!r}")
            return key  # Passthrough
        return value

    # --- Komfort: Fabrik zum Erzeugen aus mutierbaren Dicts mit defensiver Kopie ---
    @classmethod
    def from_dicts(
        cls,
        broker_map: MutableMapping[str, str],
        data_map: Optional[MutableMapping[str, str]] = None,
        *,
        strict_missing: bool = False,
    ) -> "SymbolMapper":
        """Defensive Kopie und Validierung; identisch zu __init__, aber mit MutableMapping-Signatur."""
        return cls(
            dict(broker_map),
            dict(data_map) if data_map is not None else None,
            strict_missing=strict_missing,
        )

    def __repr__(self) -> str:
        return (
            f"SymbolMapper(broker_map={dict(self._broker_map)!r}, "
            f"data_map={dict(self._data_map)!r}, strict_missing={self._cfg.strict_missing})"
        )
