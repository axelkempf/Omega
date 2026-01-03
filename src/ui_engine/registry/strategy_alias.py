# engine_ui/registry/strategy_alias.py

# MAPPING: logischer Name → technische ID
STRATEGY_ALIAS = {
    "datafeed": "15582434",
    "dxfeed": "dxfeed01",
    "account_10928521": "10928521",
    "account_10929345": "10929345",
    "account_10927144": "10927144",
}


def resolve_alias(alias: str) -> str:
    return STRATEGY_ALIAS.get(alias, alias)
