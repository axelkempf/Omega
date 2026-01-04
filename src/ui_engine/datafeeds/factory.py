from ui_engine.datafeeds.dxfeed_manager import DXFeedDatafeedManager
from ui_engine.datafeeds.mt5_manager import MT5DatafeedManager
from ui_engine.datafeeds.base import BaseDatafeedManager
from ui_engine.registry.strategy_alias import resolve_alias

# Mapping nur für echte Datafeeds
DATAFEED_PROVIDER = {
    "datafeed": "MT5",
    "dxfeed": "DXFEED",
}


def get_datafeed_manager(alias: str) -> BaseDatafeedManager:
    resolved = resolve_alias(alias)

    # ✅ Sonderbehandlung für echte Datafeeds
    if alias in DATAFEED_PROVIDER:
        provider = DATAFEED_PROVIDER[alias]
        if provider == "MT5":
            return MT5DatafeedManager(alias, resolved)
        elif provider == "DXFEED":
            return DXFeedDatafeedManager(alias, resolved)
        else:
            raise ValueError(f"Unbekannter Datafeed-Typ: {provider}")

    # ✅ Alle anderen sind Strategien, ebenfalls via MT5Manager
    return MT5DatafeedManager(alias, resolved)
