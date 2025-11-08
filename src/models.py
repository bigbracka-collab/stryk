# src/stryktips/models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# En match på kupongen
@dataclass
class StryktipsMatch:
    idx: int                 # 1..13
    home: str
    away: str
    start_time: Optional[datetime] = None
    # odds: decimalodds (1, X, 2)
    odds_1: Optional[float] = None
    odds_X: Optional[float] = None
    odds_2: Optional[float] = None
    # AI-prob (H/D/A), summerar ~1
    p1: Optional[float] = None
    pX: Optional[float] = None
    p2: Optional[float] = None

    def value_vector(self) -> Dict[str, float]:
        v = {}
        if all([self.p1, self.odds_1]):
            v["1"] = self.p1 * self.odds_1 - 1
        if all([self.pX, self.odds_X]):
            v["X"] = self.pX * self.odds_X - 1
        if all([self.p2, self.odds_2]):
            v["2"] = self.p2 * self.odds_2 - 1
        return v

# En raddefinition för matchen: singel/halv/hel
@dataclass
class Guardering:
    # outcomes ⊆ {"1","X","2"}; längd 1 (singel), 2 (halv), 3 (hel)
    outcomes: Tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.outcomes)

# Ett system (13 matcher)
@dataclass
class StryktipsSystem:
    matches: List[StryktipsMatch]
    choices: Dict[int, Guardering]   # key: match.idx

    row_cost: float = 1.0            # konfig: radpris (parametrera)
    def rows(self) -> int:
        r = 1
        for i in range(1, 14):
            r *= self.choices[i].width
        return r

    def cost(self) -> float:
        return self.rows() * self.row_cost

    def to_rows(self) -> List[str]:
        """Expandera systemet till en lista av rader (sträng med 13 tecken)."""
        # enkel rekursiv expansion
        def expand(i: int, prefix: List[str]) -> List[str]:
            if i > 13:
                return ["".join(prefix)]
            outs = self.choices[i].outcomes
            rows = []
            for o in outs:
                rows += expand(i + 1, prefix + [o])
            return rows
        return expand(1, [])

