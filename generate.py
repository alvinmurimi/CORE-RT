"""CORE-RT synthetic generator. Builds timestamp-free, novel-string cases whose gold is fixed by a
HIDDEN provenance record, not by any surface cue a resolver could sort on. See SPEC.md.

Every assertion is a bare present-tense sentence ("The {attr} of {entity} is {value}."). The deciding
retraction references a VALUE, never a position, so the whole pile can be shuffled and gold stays
decorrelated from ingestion order. The poison value is asserted MORE often than the truth (and restated
by distractors), so a frequency vote picks the wrong answer. The only path to gold is understanding the
retraction.

Output: one JSON object per line with
  memories     : list[str]              shuffled assertion texts (what a system sees)
  provenance   : list[dict]             aligned to memories; HIDDEN (scorer only)
  gold         : {"type":"value","value":...} | {"type":"abstain"}
  values       : {"v_true","v_poison","v_new"} known-value set for the deterministic scorer
plus bookkeeping (family, k_poison, n_distractors, negation_only, seed, entity, attribute, question).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SYL = ("tar von mal quen bryn sed wol kor pyx zel dree fenn gorm hask ilv jarr lome nyx orr pell rund "
       "sval threx ulm vask wend yarn zoth brae cind dorn esk fyl grau hev ish kael lorn myr oss prmagg "
       "quoll rhad skel tius uvarn vohl wyrm xil yaric zonn").split()

ORG_TAIL = ["Compact", "Authority", "Trust", "Directorate", "Bureau", "Consortium", "Syndicate",
            "Office", "Assembly", "Charter", "Foundation", "Council"]
ORG_TAIL2 = ["Holdings", "Partners", "Industries", "Group", "Systems", "Works", "Collective", "Union"]

ATTRS = ["operating sponsor", "designated custodian", "controlling office", "assigned steward",
         "lead underwriter", "governing trustee", "oversight authority", "primary guarantor",
         "appointed liquidator", "registered proprietor"]

# Cue paraphrase bank, split TRAIN/TEST. Eval cases draw ONLY from TEST (held-out), so a keyword parser
# tuned on TRAIN cues gets no free lift. Negation-only kills the poison without naming the truth.
RETRACT_NEG = [
    "An earlier note recording that the {attr} of {entity} is {vp} was filed in error and withdrawn.",
    "The claim that the {attr} of {entity} is {vp} proved mistaken and should be disregarded.",
    "Correction: {entity} does not have {vp} as its {attr}; that entry was a mix-up.",
    "It has since been established that naming {vp} as the {attr} of {entity} was incorrect.",
    "The record listing {vp} as the {attr} of {entity} has been retracted as erroneous.",
    "Disregard the report that the {attr} of {entity} is {vp}; it was logged by mistake.",
]
RETRACT_RESTATE = [
    "An earlier note naming {vp} as the {attr} of {entity} was an error; the {attr} is in fact {vt}.",
    "Correction: the {attr} of {entity} was wrongly given as {vp}; it is actually {vt}.",
    "The entry stating the {attr} of {entity} is {vp} was mistaken; the correct value is {vt}.",
    "{vp} was recorded in error as the {attr} of {entity}; that role belongs to {vt}.",
]
REUPDATE = [
    "The {attr} of {entity} has since been validly reassigned to {vn}.",
    "Following the correction, {entity} now has {vn} as its {attr}.",
]


def _split(bank: list[str], test: bool) -> list[str]:
    half = (len(bank) + 1) // 2
    return bank[half:] if test else bank[:half]


class Namer:
    """Per-case novel-string factory with collision avoidance."""

    def __init__(self, rng: random.Random):
        self.rng, self.used = rng, set()

    def _w(self, n: int) -> str:
        return "".join(self.rng.choice(SYL) for _ in range(n)).capitalize()

    def _fresh(self, fn) -> str:
        for _ in range(50):
            s = fn()
            if s not in self.used:
                self.used.add(s)
                return s
        self.used.add(s)
        return s

    def person(self) -> str:
        return self._fresh(lambda: f"{self._w(2)} {self._w(2)}")

    def org(self) -> str:
        def make():
            if self.rng.random() < 0.6:
                return f"the {self._w(2)} {self.rng.choice(ORG_TAIL)}"
            return f"{self._w(2)} {self.rng.choice(ORG_TAIL2)}"
        return self._fresh(make)


def _assert(attr: str, entity: str, value: str) -> str:
    return f"The {attr} of {entity} is {value}."


def _stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _distractors(rng: random.Random, nm: Namer, entity: str, attr: str, poison_val: str | None,
                 n: int) -> list[tuple[str, dict]]:
    """Burial: same-entity/other-attribute lines, verbatim restatements of the poison value (so a
    frequency vote loses), and same-attribute retractions about OTHER entities (lexical near-duplicates
    of the deciding retraction). None of them assert the queried (entity, attr)."""
    other_attrs = [a for a in ATTRS if a != attr]
    out: list[tuple[str, dict]] = []
    for _ in range(n):
        roll = rng.random()
        if poison_val and roll < 0.30:
            # restate the poison value verbatim, on a DIFFERENT attribute of the same entity
            oa = rng.choice(other_attrs)
            out.append((_assert(oa, entity, poison_val),
                        {"role": "distractor", "value": poison_val, "kills": None, "subj": entity, "attr": oa}))
        elif roll < 0.62:
            # same entity, different attribute, fresh value
            oa = rng.choice(other_attrs)
            v = nm.org()
            out.append((_assert(oa, entity, v),
                        {"role": "distractor", "value": v, "kills": None, "subj": entity, "attr": oa}))
        else:
            # a same-attribute retraction about ANOTHER entity (vocabulary overlap, irrelevant to gold)
            oe = nm.person()
            ov = nm.org()
            tmpl = rng.choice(_split(RETRACT_NEG, test=True))
            out.append((tmpl.format(attr=attr, entity=oe, vp=ov),
                        {"role": "distractor", "value": ov, "kills": ov, "subj": oe, "attr": attr}))
    return out


def build_case(family: str, idx: int, k: int, n_distractors: int, seed: int,
               negation_only: bool) -> dict:
    rng = random.Random(_stable_seed(seed, family, idx, k, n_distractors, negation_only))
    nm = Namer(rng)
    entity = nm.person()
    attr = rng.choice(ATTRS)
    v_true = nm.org()
    v_poison = nm.org()
    v_new = nm.org()
    question = f"What is the {attr} of {entity}?"
    core: list[tuple[str, dict]] = []
    gold: dict
    values = {"v_true": v_true, "v_poison": v_poison, "v_new": v_new}

    if family == "A":
        core.append((_assert(attr, entity, v_true),
                     {"role": "original", "value": v_true, "kills": None, "subj": entity, "attr": attr}))
        for _ in range(k):
            core.append((_assert(attr, entity, v_poison),
                         {"role": "poison", "value": v_poison, "kills": None, "subj": entity, "attr": attr}))
        if negation_only:
            tmpl = rng.choice(_split(RETRACT_NEG, test=True))
            core.append((tmpl.format(attr=attr, entity=entity, vp=v_poison),
                         {"role": "retraction", "value": None, "kills": v_poison, "subj": entity, "attr": attr}))
        else:
            tmpl = rng.choice(_split(RETRACT_RESTATE, test=True))
            core.append((tmpl.format(attr=attr, entity=entity, vp=v_poison, vt=v_true),
                         {"role": "retraction", "value": v_true, "kills": v_poison, "subj": entity, "attr": attr}))
        gold = {"type": "value", "value": v_true}

    elif family == "B":  # no-correction control: dominant value is the honest answer
        core.append((_assert(attr, entity, v_true),
                     {"role": "original", "value": v_true, "kills": None, "subj": entity, "attr": attr}))
        for _ in range(max(2, k)):
            core.append((_assert(attr, entity, v_poison),
                         {"role": "poison", "value": v_poison, "kills": None, "subj": entity, "attr": attr}))
        gold = {"type": "value", "value": v_poison}

    elif family == "C":  # unresolvable -> abstain
        if rng.random() < 0.5:  # two comparable values, no retraction, no tiebreaker
            for _ in range(max(1, k // 2) or 1):
                core.append((_assert(attr, entity, v_true),
                             {"role": "conflict", "value": v_true, "kills": None, "subj": entity, "attr": attr}))
                core.append((_assert(attr, entity, v_poison),
                             {"role": "conflict", "value": v_poison, "kills": None, "subj": entity, "attr": attr}))
        else:  # two contradictory retractions -> both killed
            core.append((_assert(attr, entity, v_true),
                         {"role": "conflict", "value": v_true, "kills": None, "subj": entity, "attr": attr}))
            core.append((_assert(attr, entity, v_poison),
                         {"role": "conflict", "value": v_poison, "kills": None, "subj": entity, "attr": attr}))
            t1 = rng.choice(_split(RETRACT_NEG, test=True))
            t2 = rng.choice(_split(RETRACT_NEG, test=True))
            core.append((t1.format(attr=attr, entity=entity, vp=v_true),
                         {"role": "retraction", "value": None, "kills": v_true, "subj": entity, "attr": attr}))
            core.append((t2.format(attr=attr, entity=entity, vp=v_poison),
                         {"role": "retraction", "value": None, "kills": v_poison, "subj": entity, "attr": attr}))
        gold = {"type": "abstain"}

    elif family == "D":  # re-update: retract poison, then valid re-assertion to v_new
        core.append((_assert(attr, entity, v_true),
                     {"role": "original", "value": v_true, "kills": None, "subj": entity, "attr": attr}))
        for _ in range(k):
            core.append((_assert(attr, entity, v_poison),
                         {"role": "poison", "value": v_poison, "kills": None, "subj": entity, "attr": attr}))
        tmpl = rng.choice(_split(RETRACT_NEG, test=True))
        core.append((tmpl.format(attr=attr, entity=entity, vp=v_poison),
                     {"role": "retraction", "value": None, "kills": v_poison, "subj": entity, "attr": attr}))
        rt = rng.choice(REUPDATE)
        core.append((rt.format(attr=attr, entity=entity, vn=v_new),
                     {"role": "reupdate", "value": v_new, "kills": None, "subj": entity, "attr": attr}))
        gold = {"type": "value", "value": v_new}
    else:
        raise ValueError(family)

    poison_for_burial = v_poison if family in {"A", "B", "D"} else None
    entries = core + _distractors(rng, nm, entity, attr, poison_for_burial, n_distractors)
    rng.shuffle(entries)
    return {
        "id": f"{family}{'-neg' if (family == 'A' and negation_only) else ''}-{idx:04d}",
        "family": family, "negation_only": bool(family == "A" and negation_only),
        "k_poison": k, "n_distractors": n_distractors, "seed": seed,
        "entity": entity, "attribute": attr, "question": question,
        "memories": [e[0] for e in entries],
        "provenance": [e[1] for e in entries],
        "gold": gold, "values": values,
    }


def build(per_family: int, k: int, n_distractors: int, seed: int) -> list[dict]:
    out: list[dict] = []
    for i in range(per_family):  # A: half negation-only, half restating
        out.append(build_case("A", i, k, n_distractors, seed, negation_only=(i % 2 == 0)))
    for fam in ("B", "C", "D"):
        for i in range(per_family):
            out.append(build_case(fam, i, k, n_distractors, seed, negation_only=False))
    random.Random(seed).shuffle(out)
    return out


def main():
    p = argparse.ArgumentParser(description="Build CORE-RT synthetic cases.")
    p.add_argument("--per-family", type=int, default=40)
    p.add_argument("--k", type=int, default=4, help="poison restatements in families A/B/D")
    p.add_argument("--distractors", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="data/cases.jsonl")
    a = p.parse_args()
    out = build(a.per_family, a.k, a.distractors, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out).open("w", encoding="utf-8") as fh:
        for c in out:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(out)} cases to {a.out}")
    print("by family:", dict(Counter(c["family"] for c in out)))
    print(f"avg memories/case: {sum(len(c['memories']) for c in out) / len(out):.0f}")


if __name__ == "__main__":
    main()
