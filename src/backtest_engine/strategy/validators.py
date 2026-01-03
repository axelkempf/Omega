import importlib
from typing import Dict, List


def validate_strategy_config(
    strat_conf: Dict[str, any], mode: str = "candle"
) -> List[str]:
    """
    Validates a single strategy configuration dictionary.

    Args:
        strat_conf (dict): Strategy configuration with keys:
            - 'module' (str): Name of the Python module (in strategies).
            - 'class' (str): Name of the strategy class.
            - 'parameters' (dict): Dict of strategy parameters (may be empty, but must exist).
            - (Optional) 'symbol' (str): Required if mode == 'tick'.
        mode (str): Mode ("candle" or "tick"). Tick mode requires a symbol.

    Returns:
        List[str]: List of error messages, empty if valid.
    """
    errors: List[str] = []

    if "module" not in strat_conf:
        errors.append("🟥 'module' fehlt in der Strategie-Konfiguration.")
    if "class" not in strat_conf:
        errors.append("🟥 'class' fehlt in der Strategie-Konfiguration.")
    if "parameters" not in strat_conf:
        errors.append("🟥 'parameters' fehlt – auch wenn leer, muss vorhanden sein.")

    # Dynamisch: Existiert Modul und Klasse?
    if "module" in strat_conf and "class" in strat_conf:
        try:
            module_path = strat_conf["module"]
            if not module_path.startswith("strategies."):
                module_path = f"strategies.{module_path}"
            mod = importlib.import_module(module_path)
            if not hasattr(mod, strat_conf["class"]):
                errors.append(
                    f"🟥 Klasse '{strat_conf['class']}' nicht im Modul '{strat_conf['module']}' gefunden."
                )
        except ModuleNotFoundError:
            errors.append(f"🟥 Modul '{module_path}' nicht gefunden.")

    # Tick-Modus: Symbol ist Pflicht
    if mode == "tick" and "symbol" not in strat_conf:
        errors.append("🟥 Im Tick-Modus muss jede Strategie ein 'symbol' enthalten.")

    return errors
