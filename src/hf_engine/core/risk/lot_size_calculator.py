"""
lot_size_calculator.py

Berechnet Pip-Werte und Lotgrößen basierend auf Risikoparametern und Brokerdaten.
Optimierungen:
- Robustes Symbol-Parsing (Base/Quote/Suffix) mit Fallbacks
- Währungsumrechnung Quote -> AccountCurrency via Cross (mit/ohne Suffix), Preisableitung robust
- Berücksichtigung von broker-spezifischer Volumenpräzision (min_volume, volume_step, max_volume)
- Präzisere Rundung mit Decimal; risikobewusste "floor"-Rundung auf volume_step
- Defensives Error-Handling mit klaren Meldungen
"""

from __future__ import annotations

import re
from decimal import ROUND_FLOOR, Decimal, InvalidOperation, getcontext
from typing import Any, Dict, Optional, Tuple

from hf_engine.adapter.broker.broker_interface import BrokerInterface
from hf_engine.adapter.broker.broker_utils import get_pip_size
from hf_engine.core.execution.sl_tp_utils import distance_to_sl
from hf_engine.infra.logging.log_service import log_service

logger = log_service.logger
getcontext().prec = 28  # hohe Präzision für Finanzrundungen


# ---- interne Utilities ----------------------------------------------------- #


def _info_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Versucht mehrere Schlüsselvarianten aus Symbolinfo; erster Treffer gewinnt."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


_SYMBOL_RE = re.compile(r"^(?P<base>[A-Z]{3})(?P<quote>[A-Z]{3})(?P<suffix>.*)$")


def _parse_symbol(symbol: str, info: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    """
    Ermittelt Base/Quote/Suffix möglichst aus Broker-Info; fällt bei Bedarf auf Regex/Heuristik zurück.
    """
    base = None
    quote = None
    suffix = ""

    if info:
        base = _info_get(
            info,
            "currency_base",
            "base_currency",
            "base",
            "symbol_base",
        )
        # 'currency_profit' ist bei vielen Brokern die Quote-/Profit-Währung
        quote = _info_get(
            info,
            "currency_profit",
            "currency_quote",
            "quote_currency",
            "quote",
            "profit_currency",
            "symbol_quote",
        )
        # Einige Broker führen das Suffix separat
        suffix_from_info = _info_get(info, "suffix", default=None)
        if isinstance(suffix_from_info, str):
            suffix = suffix_from_info

    if not base or not quote:
        m = _SYMBOL_RE.match(symbol.upper())
        if m:
            base = base or m.group("base")
            quote = quote or m.group("quote")
            # Suffix aus dem Symbol nur nutzen, wenn es nicht bereits aus info kam
            suffix = suffix or m.group("suffix")
        else:
            # minimaler Fallback: klassische 3+3-Aufteilung
            if len(symbol) >= 6:
                base = base or symbol[:3].upper()
                quote = quote or symbol[3:6].upper()
                suffix = suffix or symbol[6:]
            else:
                raise ValueError(
                    f"❌ Symbol '{symbol}' kann nicht geparst werden (erwartet mind. 6 Zeichen)."
                )

    return base, quote, suffix or ""


def _price_from_tick(tick: Dict[str, Any]) -> Optional[float]:
    """
    Extrahiert einen fairen Preis aus einem Tick:
    bevorzuge Mid (bid/ask), dann bid, dann last/price/ask.
    """
    try:
        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
        if bid is not None:
            return float(bid)
        if tick.get("last") is not None:
            return float(tick["last"])
        if tick.get("price") is not None:
            return float(tick["price"])
        if ask is not None:
            return float(ask)
    except Exception as e:
        logger.debug("Tick-Preis konnte nicht extrahiert werden: %s | tick=%s", e, tick)
    return None


def _round_volume_to_step(
    volume: float,
    step: float,
    rounding_mode: str = "floor",
) -> float:
    """
    Rundet die Volumenangabe auf die broker-spezifische Schrittweite.
    Standard: FLOOR (risikobewusst, überschätzt Risiko nicht).
    """
    try:
        v = Decimal(str(volume))
        s = Decimal(str(step)) if step else Decimal("0.01")
        if s <= 0:
            s = Decimal("0.01")
        if rounding_mode == "ceil":
            q = (v / s).to_integral_value(rounding="ROUND_CEILING")
        elif rounding_mode == "nearest":
            q = (v / s).to_integral_value(rounding="ROUND_HALF_UP")
        else:  # floor
            q = (v / s).to_integral_value(rounding=ROUND_FLOOR)
        return float(q * s)
    except (InvalidOperation, ValueError) as e:
        logger.warning(
            "Volumenrundung fehlgeschlagen (%s). Fallback auf 2 Dezimalstellen.", e
        )
        return round(volume, 2)


def _clamp(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is not None:
        value = max(value, low)
    if high is not None:
        value = min(value, high)
    return value


# ---- öffentliche API ------------------------------------------------------- #


def get_price_spec(symbol: str, broker: BrokerInterface) -> Dict[str, float]:
    info = broker.get_symbol_info(symbol) or {}
    # point := kleinste Preisinkrement, bevorzugt tick_size, sonst aus digits ableiten
    tick_size = float(info.get("tick_size") or 0.0)
    if tick_size <= 0.0:
        digits = int(info.get("digits") or 5)
        tick_size = 10.0 ** (-digits)
    return {
        "point": tick_size,
        "tick_size": tick_size,
        "tick_value": float(info.get("tick_value") or 0.0),
        "contract_size": float(info.get("contract_size") or 0.0),
        "quote_ccy": info.get("currency_profit"),
        "volume_min": float(info.get("volume_min") or 0.01),
        "volume_step": float(info.get("volume_step") or 0.01),
        "volume_max": float(info.get("volume_max") or 0.0),
    }


def point_value(symbol: str, lot_size: float, broker: BrokerInterface) -> float:
    """Wert eines minimalen Preisinkrements (point) in Account-Währung je Lot."""
    spec = get_price_spec(symbol, broker)
    # MT5 liefert oft tick_value je 1 Lot bereits in Account-Währung – falls vorhanden, nutze direkt.
    if spec["tick_value"] > 0:
        return spec["tick_value"] * float(lot_size)
    # Fallback (FX/CFDs): pip/punkt-Wert aus contract_size * point, dann Quote→Account konvertieren
    info = broker.get_symbol_info(symbol)
    contract_size = float(info.get("contract_size") or 0.0)
    pv_quote = spec["point"] * contract_size * float(lot_size)
    # Quote→Account via Cross (bestehende Logik wiederverwenden)
    return get_pip_value(symbol, lot_size, broker) * (
        spec["point"] / get_pip_size(symbol)
    )


def get_pip_value(symbol: str, lot_size: float, broker: BrokerInterface) -> float:
    """
    Berechnet den Wert eines Pips in Kontowährung für ein gegebenes Symbol und eine Lotgröße.

    Formel (Spot/CFD typisch):
        pip_value_quote_ccy = pip_size * contract_size * lot_size
        pip_value_account   = pip_value_quote_ccy * FX(quote -> account)

    Dabei wird ein Cross gesucht:
        QUOTE + ACCOUNT (+ ggf. Suffix)  -> Multiplikation
        ACCOUNT + QUOTE (+ ggf. Suffix)  -> Division (Inverse)

    Raises:
        RuntimeError | ValueError bei fehlenden Infos/Konvertierungen.
    """
    info = broker.get_symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"❌ Symbolinfo für {symbol} nicht verfügbar.")

    account_currency = broker.get_account_currency()
    pip_size = get_pip_size(symbol)
    contract_size = _info_get(
        info, "contract_size", "volume_contract_size", default=None
    )

    if contract_size is None:
        raise ValueError(f"❌ 'contract_size' für {symbol} fehlt in Symbolinfo.")

    try:
        contract_size = float(contract_size)
    except Exception as e:
        raise ValueError(
            f"❌ 'contract_size' für {symbol} ungültig: {contract_size} ({e})"
        )

    pip_value_quote = pip_size * contract_size * float(lot_size)

    base, quote, suffix = _parse_symbol(symbol, info)

    if not account_currency:
        raise RuntimeError("❌ Konto-Währung konnte nicht ermittelt werden.")

    # Wenn Quote bereits Kontowährung ist, keine Umrechnung nötig
    if quote == account_currency:
        return pip_value_quote

    # Kandidatenreihenfolge für Cross-Symbole: zuerst mit identischem Suffix, dann ohne Suffix
    candidates = [
        (
            f"{account_currency}{quote}{suffix}"
            if suffix
            else f"{account_currency}{quote}"
        ),
        (
            f"{quote}{account_currency}{suffix}"
            if suffix
            else f"{quote}{account_currency}"
        ),
    ]
    # Falls kein Suffix im ursprünglichen Symbol erkannt, versuche zusätzlich ohne/mit generischem Suffix nicht – Broker-spezifisch.

    price_used = None
    for cross in candidates:
        try:
            tick = broker.get_symbol_tick(cross)
        except Exception:
            # still & quick: einfach nächsten Kandidaten versuchen
            tick = None

        if not tick:
            # Falls mit Suffix nicht verfügbar war, versuche ohne Suffix (nur wenn vorher mit Suffix versucht)
            if suffix and cross.endswith(suffix):
                alt = cross[: -len(suffix)]
                try:
                    tick = broker.get_symbol_tick(alt)
                    if tick:
                        logger.debug(
                            "Fiel auf Cross ohne Suffix zurück: %s -> %s", cross, alt
                        )
                        cross = alt
                except Exception as e:
                    logger.debug(
                        "Fehler bei Tick-Abruf (ohne Suffix) für Cross %s: %s", alt, e
                    )
                    tick = None

        price = _price_from_tick(tick) if tick else None
        if price is None:
            continue

        price_used = price
        if cross.startswith(quote):  # QUOTE/ACCOUNT -> multiplizieren
            return pip_value_quote * price_used
        else:  # ACCOUNT/QUOTE -> dividieren (Inverse)
            # Schutz vor Division durch 0
            if price_used <= 0:
                continue
            return pip_value_quote / price_used

    raise ValueError(
        f"⚠️ Keine Umrechnung möglich: {symbol} ({quote}) → Konto-Währung {account_currency} "
        f"[getestete Crosses: {', '.join(candidates)}]"
    )


def calculate_lot_size(setup: Any, broker: BrokerInterface) -> float:
    """
    Berechnet die optimale Lotgröße auf Basis des Risiko-Setups.

    Erwartete Felder in 'setup':
        - symbol: str
        - entry: float
        - sl: float
        - start_capital: float
        - risk_pct: float  (z. B. 1.0 für 1%)

    Broker-Parameter (aus Symbolinfo), falls verfügbar:
        - min_volume | volume_min
        - volume_step | lot_step
        - max_volume | volume_max

    Rundungsstrategie:
        - Lot wird nach Berechnung "risikobewusst" auf den nächsten Schritt NACH UNTEN gerundet.
        - Danach Clamp auf [min_volume, max_volume].
    """
    symbol = setup.symbol
    info = broker.get_symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"❌ Symbolinfo für {symbol} nicht verfügbar.")

    # Statische Größen
    spec = get_price_spec(symbol, broker)
    point = float(spec["point"])
    if point <= 0:
        raise ValueError(f"❌ Ungültige Tick/Point-Größe für {symbol}: {point}")

    pv_1lot = point_value(symbol, lot_size=1.0, broker=broker)
    if pv_1lot <= 0:
        raise ValueError(f"❌ Ungültiger Pip-Wert für {symbol}: {pv_1lot}")

    # SL-Distanz (in Preis & Pips)
    sl_dist_price = distance_to_sl(setup.entry, setup.sl)
    sl_points = sl_dist_price / point if point > 0 else 0.0
    if sl_points <= 0:
        raise ValueError(
            f"❌ SL-Distanz darf nicht 0/negativ sein ({symbol}). entry={setup.entry}, sl={setup.sl}"
        )

    # Risiko in Kontowährung
    try:
        start_capital = float(setup.start_capital)
        risk_pct = float(setup.risk_pct)
    except Exception as e:
        raise ValueError(f"❌ Ungültige Setup-Werte (start_capital/risk_pct): {e}")

    if start_capital <= 0 or risk_pct <= 0:
        raise ValueError(
            f"❌ start_capital und risk_pct müssen > 0 sein (start_capital={start_capital}, risk_pct={risk_pct})."
        )

    risk_amount = start_capital * (risk_pct / 100.0)

    # Theoretische Lotgröße (vor Rundung)
    lot_theoretical = risk_amount / (pv_1lot * sl_points)

    # Broker-Volumenparameter
    min_vol = _info_get(info, "volume_min", "min_volume", default=0.01)
    step_vol = _info_get(info, "volume_step", "lot_step", default=0.01)
    max_vol = _info_get(info, "volume_max", "max_volume", default=None)

    try:
        min_vol = float(min_vol) if min_vol is not None else 0.01
        step_vol = float(step_vol) if step_vol is not None else 0.01
        max_vol = float(max_vol) if max_vol is not None else None
    except Exception as e:
        logger.warning(
            "Ungültige Volumenparameter in Symbolinfo (%s). Fallback auf (min=0.01, step=0.01).",
            e,
        )
        min_vol, step_vol, max_vol = 0.01, 0.01, None

    # Risikobewusste Rundung auf Volumenstep (floor)
    lot_rounded = _round_volume_to_step(
        lot_theoretical, step_vol, rounding_mode="floor"
    )

    # Clamp auf broker-Grenzen
    lot_final = _clamp(lot_rounded, min_vol, max_vol)

    # Hinweis: Falls floor-Rundung unter min_vol liegt, wird auf min_vol geklemmt -> Risiko kann leicht über Ziel liegen.
    # Das ist in Live-Umgebungen üblich, alternativ könnte man in solchen Fällen den Trade verwerfen.

    if lot_final <= 0:
        raise ValueError(
            f"❌ Ergebnis-Lotgröße ist 0 oder negativ (theoretisch={lot_theoretical:.8f}, gerundet={lot_rounded:.8f}). "
            f"Prüfe Risiko/SL/Instrument."
        )

    return float(lot_final)
