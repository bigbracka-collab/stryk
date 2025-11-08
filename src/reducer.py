# src/stryktips/reducer.py
from __future__ import annotations
from typing import List
from .models import StryktipsSystem

def filter_rows_by_upsets(rows: List[str], upset_min: int = 0, upset_max: int = 13, fav_symbol: str = "1") -> List[str]:
    """
    Naiv reducering: filtrera rader baserat på hur många "icke-favorit" tecken de har.
    Här definierat som allt ≠ fav_symbol, men du kan förbättra med dynamisk favorit per match.
    """
    out = []
    for r in rows:
        # r är str med 13 tecken, t.ex. "1X21X..." – räkna "skrällar"
        upsets = sum(1 for c in r if c != fav_symbol)
        if upset_min <= upsets <= upset_max:
            out.append(r)
    return out
