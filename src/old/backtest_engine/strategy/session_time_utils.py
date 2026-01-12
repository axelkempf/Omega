from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Tuple

import pytz

# Zeitzonen für Anker-Sessions (UTC-Automation für institutionelle Märkte)
timezones: Dict[str, Any] = {
    "london": pytz.timezone("Europe/London"),
    "new_york": pytz.timezone("America/New_York"),
    "tokyo": pytz.timezone("Asia/Tokyo"),
    "sydney": pytz.timezone("Australia/Sydney"),
    "frankfurt": pytz.timezone("Europe/Berlin"),
    "asia": pytz.timezone("Asia/Tokyo"),  # Alias für Asien-Sessions
    "europe": pytz.timezone("Europe/London"),  # Alias für Europa
}

# Lokale Session-Startzeiten (pro Session, für DST-awareness)
SESSION_START_LOCAL: Dict[str, Tuple[int, int]] = {
    "london": (8, 0),
    "new_york": (9, 30),
    "tokyo": (9, 0),
    "sydney": (8, 0),
    "frankfurt": (8, 0),
    "asia": (9, 0),  # alias tokyo
    "europe": (8, 0),  # alias london/frankfurt
}


def parse_offset(offset_str: str) -> timedelta:
    """
    Parses a UTC offset string (e.g., '-03:00') to a timedelta object.

    Args:
        offset_str (str): Offset as string, e.g. '-03:00', '+02:00'.

    Returns:
        timedelta: Corresponding timedelta object.
    """
    sign = -1 if offset_str.startswith("-") else 1
    h, m = map(int, offset_str.strip("+-").split(":"))
    return timedelta(hours=h, minutes=m) * sign


def generate_anchored_windows(
    start_date: str,
    end_date: str,
    anchor: str,
    offset_window: List[Tuple[str, str]],
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Creates session windows anchored to a trading session (e.g., London),
    with automatic handling of DST and local session shifts.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        anchor (str): Anchor session name (must exist in `timezones` and `SESSION_START_LOCAL`).
        offset_window (List[Tuple[str, str]]): List of (start, end) time offsets as strings, e.g. [("-03:00", "-01:00")].

    Returns:
        Dict[str, List[Tuple[str, str]]]: Mapping date string ('YYYY-MM-DD') to list of (start, end) session windows in UTC time ('HH:MM').
    """
    if anchor not in timezones:
        raise ValueError(f"Unknown anchor: {anchor}")

    tz = timezones[anchor]
    start_hour, start_minute = SESSION_START_LOCAL[anchor]
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    one_day = timedelta(days=1)

    result: Dict[str, List[Tuple[str, str]]] = {}

    current = start
    while current <= end:
        d = current.date()
        # Localize anchor to correct local time, auto-handling DST
        anchor_local = tz.localize(
            datetime(d.year, d.month, d.day, start_hour, start_minute)
        )
        windows: List[Tuple[str, str]] = []

        for offset_start_str, offset_end_str in offset_window:
            offset_start = parse_offset(offset_start_str)
            offset_end = parse_offset(offset_end_str)

            window_start = (anchor_local + offset_start).astimezone(pytz.UTC).time()
            window_end = (anchor_local + offset_end).astimezone(pytz.UTC).time()
            windows.append(
                (window_start.strftime("%H:%M"), window_end.strftime("%H:%M"))
            )

        result[str(d)] = windows
        current += one_day

    return result


def generate_anchored_windows_combined(
    start_date: str,
    end_date: str,
    windows: List[Dict[str, Any]],
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Combines multiple anchored session window configs into a single calendar.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        windows (List[Dict]): Each dict must have "anchor" and "offset_window" keys.

    Returns:
        Dict[str, List[Tuple[str, str]]]: Combined session window calendar.
    """
    calendar: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for window in windows:
        anchor = window["anchor"]
        offsets = window["offset_window"]
        single_calendar = generate_anchored_windows(
            start_date, end_date, anchor, offsets
        )
        for date_str, periods in single_calendar.items():
            calendar[date_str].extend(periods)
    return dict(calendar)
