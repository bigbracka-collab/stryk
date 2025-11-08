# src/stryktips/generator.py
from __future__ import annotations
from typing import List, Dict, Tuple
from .models import StryktipsMatch, Guardering, StryktipsSystem

def pick_guardering_for_match(m: StryktipsMatch,
                              prefer_value: bool = True,
                              min_edge: float = 0.02,
                              max_width: int = 3) -> Guardering:
    """
    Heuristik:
    - Om tydlig favorit (p > 0.55) och value ≥ min_edge -> spika (singel)
    - Om två alternativ sticker ut i prob/value -> halvgardering
    - Annars helgardera (upp till max_width)
    """
    # sortera på value primärt, annars prob
    v = m.value_vector()
    triplets = []
    for k, p in [("1", m.p1 or 0), ("X", m.pX or 0), ("2", m.p2 or 0)]:
        edge = v.get(k, p - 1/( (m.odds_1 if k=="1" else m.odds_X if k=="X" else m.odds_2) or 1e9 ))
        triplets.append((k, p, edge))
    # sortera: value först, fallback prob
    if prefer_value and len(v) == 3:
        triplets.sort(key=lambda x: (x[2], x[1]), reverse=True)
    else:
        triplets.sort(key=lambda x: x[1], reverse=True)

    # besluta bredd
    top = triplets[0]
    if top[1] >= 0.55 and top[2] >= min_edge:
        return Guardering((top[0],))  # spik

    # om top2 är någorlunda nära -> halv
    if len(triplets) >= 2:
        second = triplets[1]
        if (top[1] - second[1]) < 0.10 or (second[2] > 0):  # nära i prob eller båda har value
            return Guardering(tuple(sorted([top[0], second[0]])))

    # annars hel (begränsa till max_width)
    base = tuple([t[0] for t in triplets[:max_width]])
    return Guardering(tuple(sorted(set(base))))

def build_system(matches: List[StryktipsMatch],
                 budget: float,
                 row_cost: float = 1.0,
                 prefer_value: bool = True,
                 min_edge: float = 0.02,
                 max_width: int = 3) -> StryktipsSystem:
    """
    Bygg ett system under en budget:
    1) börja med heuristiska val per match
    2) om för dyrt: krymp bredd där det gör minst ont (lägst marginal-value)
    3) om för billigt: bredda med nästa bästa utfall tills budget nås (soft)
    """
    # start: heuristik
    choices = {}
    for m in matches:
        choices[m.idx] = pick_guardering_for_match(m, prefer_value, min_edge, max_width)

    sys = StryktipsSystem(matches=matches, choices=choices, row_cost=row_cost)

    # helper för att rangordna "bredd-kandidater" (vilka utfall läggs till/tas bort)
    def candidate_add_list():
        cands = []
        for m in matches:
            current = set(sys.choices[m.idx].outcomes)
            # ranka nästa bästa utfall att lägga till
            ranking = sorted(
                [("1", m.p1 or 0), ("X", m.pX or 0), ("2", m.p2 or 0)],
                key=lambda x: x[1], reverse=True
            )
            for k, p in ranking:
                if k not in current and len(current) < max_width:
                    cands.append((m.idx, k, p))
                    break
        # högst p först
        cands.sort(key=lambda x: x[2], reverse=True)
        return cands

    def candidate_drop_list():
        cands = []
        for m in matches:
            outs = list(sys.choices[m.idx].outcomes)
            if len(outs) <= 1:
                continue
            # ta bort sämsta utfall (lägst p)
            ranking = sorted(outs, key=lambda k: {"1": m.p1 or 0, "X": m.pX or 0, "2": m.p2 or 0}[k])
            cands.append((m.idx, ranking[0]))
        # ta bort lägst p först (billigast att tappa)
        return cands

    # krymp tills <= budget
    while sys.cost() > budget:
        drops = candidate_drop_list()
        if not drops:
            break
        idx, k = drops[0]
        new = tuple(o for o in sys.choices[idx].outcomes if o != k)
        sys.choices[idx] = Guardering(new)

    # om luft kvar: bredda lite mot budget (frivilligt)
    safety_rounds = 64
    while sys.cost() + row_cost <= budget and safety_rounds > 0:
        adds = candidate_add_list()
        if not adds:
            break
        idx, k, _ = adds[0]
        new = tuple(sorted(set(sys.choices[idx].outcomes) | {k}))
        sys.choices[idx] = Guardering(new)
        safety_rounds -= 1

    return sys
