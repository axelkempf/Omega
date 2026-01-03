from typing import Optional


class ExecutionResult:
    """
    Leichtgewichtiger Container für Ausführungsergebnisse.

    Attributes:
        success (bool): True bei Erfolg, sonst False. Wird defensiv zu bool gecastet.
        message (str): Freitext-Nachricht; None wird zu "" normalisiert.
        order (Optional[int]): Optionale Order-ID; nicht-konvertierbare Werte werden zu None.
    """

    def __init__(self, success: bool, message: str = "", order: Optional[int] = None):
        # Kompatibilität: gleiche Signatur, gleiche Attributnamen
        self.success: bool = bool(success)
        self.message: str = "" if message is None else str(message)

        if order is None or isinstance(order, int):
            self.order: Optional[int] = order
        else:
            # Defensiv: versuche numerische Strings o.ä. zu int zu casten, sonst None
            try:
                self.order = int(order)
            except (TypeError, ValueError):
                self.order = None

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        # Kompatibilität: unverändert zur bisherigen Darstellung
        return f"ExecutionResult(success={self.success}, message='{self.message}')"
