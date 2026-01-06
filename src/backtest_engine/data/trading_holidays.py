from __future__ import annotations

from typing import List, Sequence

import holidays


def get_holiday_calendar(
    symbol: str, years: Sequence[int] | range = range(2000, 2031)
) -> holidays.HolidayBase:
    """
    Gibt einen kombinierten Holiday-Kalender für ein FX-Symbol zurück.

    Args:
        symbol: FX-Symbol (z.B. 'EURUSD', 'GBPJPY').
        years: Jahre, für die Feiertage geladen werden sollen (default: 2000–2030).

    Returns:
        Kombinierter holidays.HolidayBase mit allen relevanten Ländern.
    """
    symbol = symbol.upper()
    country_calendars: List[holidays.HolidayBase] = []

    year_list = list(years)  # Explizit Listen-Objekt

    # Mapping Währungs-Code zu Kalender
    if "EUR" in symbol:
        country_calendars.append(holidays.ECB(years=year_list))  # type: ignore[attr-defined]
    if "USD" in symbol:
        country_calendars.append(holidays.US(years=year_list))  # type: ignore[attr-defined]
    if "GBP" in symbol:
        country_calendars.append(holidays.UK(years=year_list))  # type: ignore[attr-defined]
    if "JPY" in symbol:
        country_calendars.append(holidays.Japan(years=year_list))  # type: ignore[attr-defined]
    if "CAD" in symbol:
        country_calendars.append(holidays.CA(years=year_list))  # type: ignore[attr-defined]
    if "CHF" in symbol:
        country_calendars.append(holidays.CH(years=year_list))  # type: ignore[attr-defined]
    if "AUD" in symbol:
        country_calendars.append(holidays.Australia(years=year_list))  # type: ignore[attr-defined]
    if "NZD" in symbol:
        country_calendars.append(holidays.NewZealand(years=year_list))  # type: ignore[attr-defined]

    # Kombinieren aller relevanten Kalender
    combined = holidays.HolidayBase()
    for cal in country_calendars:
        combined.update(cal)  # type: ignore[arg-type]

    return combined
