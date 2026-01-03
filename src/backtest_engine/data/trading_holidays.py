from typing import List

import holidays


def get_holiday_calendar(
    symbol: str, years: range = range(2000, 2031)
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

    years = list(years)  # Explizit Listen-Objekt

    # Mapping Währungs-Code zu Kalender
    if "EUR" in symbol:
        country_calendars.append(holidays.EuropeanCentralBank(years=years))
    if "USD" in symbol:
        country_calendars.append(holidays.US(years=years))
    if "GBP" in symbol:
        country_calendars.append(holidays.UK(years=years))
    if "JPY" in symbol:
        country_calendars.append(holidays.Japan(years=years))
    if "CAD" in symbol:
        country_calendars.append(holidays.CA(years=years))
    if "CHF" in symbol:
        country_calendars.append(holidays.CH(years=years))
    if "AUD" in symbol:
        country_calendars.append(holidays.Australia(years=years))
    if "NZD" in symbol:
        country_calendars.append(holidays.NewZealand(years=years))

    # Kombinieren aller relevanten Kalender
    combined = holidays.HolidayBase()
    for cal in country_calendars:
        combined.update(cal)

    return combined
