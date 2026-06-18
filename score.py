"""Deterministic scorer + tautology canaries for CORE-RT.

The scorer never uses an LLM: it substring-matches a system's answer against the case's known values
({v_true, v_poison, v_new}) and an abstention vocabulary, and grades against provenance-gold. An LLM
judge may be added for phrasing tolerance ONLY if it agrees with this matcher on a hand-audited sample.

The canaries are deterministic non-LLM resolvers (most-recent-by-order, most-frequent, last-line) that
see only the plain cell assertions. On family A they MUST floor at/below chance; if any canary reaches
gold, the cell is recency/frequency-solvable and is discarded. This is the anti-tautology gate.
"""

from __future__ import annotations

import re
from collections import Counter

_ABSTAIN = ["unresolved", "not available", "unavailable", "cannot be determined", "cannot determine",
            "can't be determined", "unknown", "no definite", "conflicting", "contradict", "unclear",
            "insufficient", "not enough", "no single", "indeterminate", "not possible to determine",
            "undetermined", "no reliable", "ambiguous", "not stated", "no current", "cannot say"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^the ", "", (s or "").strip().lower())).strip(" .")


def has_value(answer: str, value: str) -> bool:
    if not value:
        return False
    return _norm(value) in _norm(answer)


def abstains(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in _ABSTAIN)


def killed_values(case: dict) -> set[str]:
    """Values explicitly retracted for the queried entity (from hidden provenance)."""
    ent = case["entity"]
    out = set()
    for p in case["provenance"]:
        if p.get("role") == "retraction" and p.get("subj") == ent and p.get("kills"):
            out.add(p["kills"])
    return out


def score_answer(case: dict, answer: str) -> dict:
    gold = case["gold"]
    vals = case["values"]
    killed = killed_values(case)
    leaked = any(has_value(answer, kv) for kv in killed)
    asserts_true = has_value(answer, vals["v_true"])
    asserts_poison = has_value(answer, vals["v_poison"])
    asserts_new = has_value(answer, vals["v_new"])
    abst = abstains(answer) and not (asserts_true or asserts_poison or asserts_new)

    if gold["type"] == "abstain":
        correct = abst
    else:
        # correct iff it states the gold value and does not also surface a retracted value
        correct = has_value(answer, gold["value"]) and not leaked
    return {"correct": bool(correct), "leaked": bool(leaked), "abstained": bool(abst),
            "asserts": {"true": asserts_true, "poison": asserts_poison, "new": asserts_new}}


# ---------- tautology canaries ----------

def _cell_assertions(case: dict) -> list[tuple[int, str]]:
    """(position, value) for memories that are a plain assertion of the queried (entity, attribute)."""
    pat = re.compile(rf"^the {re.escape(case['attribute'])} of {re.escape(case['entity'])} is (.+?)\.?$",
                     re.IGNORECASE)
    out = []
    for i, m in enumerate(case["memories"]):
        mm = pat.match(m.strip())
        if mm:
            out.append((i, mm.group(1).strip()))
    return out


def canary_recency(case: dict) -> str | None:
    a = _cell_assertions(case)
    return max(a, key=lambda t: t[0])[1] if a else None


def canary_frequency(case: dict) -> str | None:
    a = _cell_assertions(case)
    if not a:
        return None
    return Counter(v for _, v in a).most_common(1)[0][0]


def canary_lastline(case: dict) -> str | None:
    last = len(case["memories"]) - 1
    for i, v in _cell_assertions(case):
        if i == last:
            return v
    return None


CANARIES = {"recency": canary_recency, "frequency": canary_frequency, "lastline": canary_lastline}


def canary_correct(case: dict, pred: str | None) -> bool:
    """A canary 'answer' is its predicted value (or nothing). Graded by the same rule as any system."""
    return score_answer(case, pred if pred else "")["correct"]
