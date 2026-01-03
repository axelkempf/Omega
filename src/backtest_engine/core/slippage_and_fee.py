import random
from typing import Optional


class SlippageModel:
    """
    Modelliert Slippage bei Orderausführungen im Backtest.

    Args:
        fixed_pips: Feste Slippage in Pips je Trade.
        random_pips: Maximale zusätzliche, zufällige Slippage in Pips (uniform [0, random_pips]).
    """

    def __init__(self, fixed_pips: float = 0.0, random_pips: float = 0.0):
        self.fixed_pips = fixed_pips
        self.random_pips = random_pips

    def apply(self, price: float, direction: str, pip_size: float = 0.0001) -> float:
        """
        Berechnet Ausführungspreis mit Slippage.

        Args:
            price: Ursprünglicher Orderpreis.
            direction: "long" oder "short".
            pip_size:   Preisinkrement pro Pip (z. B. 0.0001 FX, 0.01 JPY/Metalle/CFDs – brokerabhängig).

        Returns:
            Preis inkl. Slippage.
        """
        slippage = self.fixed_pips + random.uniform(0, self.random_pips)
        if direction == "long":
            return price + slippage * pip_size
        else:
            return price - slippage * pip_size


class FeeModel:
    """
    Modelliert Handelsgebühren im Backtest.

    Args:
        per_million: Kommission pro 1 Mio Notional in Quote-CCY.
        lot_size:    Fallback-Contract-Size (FX=100k), falls keine symbol-spezifische Größe übergeben wird.
        min_fee:     Optionale Mindestgebühr pro Transaktion.
    """

    def __init__(
        self,
        per_million: float = 5.0,
        lot_size: float = 100_000.0,
        min_fee: float = 0.0,
    ):
        self.per_million = float(per_million)
        self.lot_size = float(lot_size)
        self.min_fee = float(min_fee)

    def calculate(
        self, volume_lots: float, price: float, contract_size: float = None
    ) -> float:
        """
        Berechnet die anfallenden Gebühren für das Volumen.

        Args:
            volume_lots: Volumen in Lots.
            price:       Preis pro Einheit (Quote-CCY).
            contract_size: Contract-Size pro Lot (falls vorhanden symbol-spezifisch).

        Returns:
            Absoluter Gebührenbetrag (gleiche Währung wie Preis).
        """
        cs = float(contract_size) if contract_size else self.lot_size
        notional = float(volume_lots) * cs * float(price)
        fee = (notional / 1_000_000.0) * self.per_million
        if self.min_fee > 0.0:
            fee = max(fee, self.min_fee)
        return float(fee)
