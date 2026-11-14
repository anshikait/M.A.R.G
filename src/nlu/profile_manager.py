"""Profile State Manager: defaults -> overrides, history, what-if copies, confirmation of big changes."""
import copy, time
from typing import Dict, List

BIG_CHANGE = {"annual_income": lambda o, n: max(o, n) / max(min(o, n), 1) >= 3,
              "age": lambda o, n: abs(o - n) >= 10}
CONFIRM_ALWAYS = {"gender", "state"}          # contradictory values for these are worth double-checking


class ProfileManager:
    def __init__(self):
        self.data: Dict[str, dict] = {}      # attr -> {value, source_turn, confidence, history:[...]}
        self.pending: Dict[str, dict] = {}
        self.turn = 0

    # ---- views
    def values(self, extra: Dict = None) -> Dict:
        v = {a: d["value"] for a, d in self.data.items()}
        v.update(extra or {})
        return v

    def whatif(self, slots: List[dict]) -> Dict:
        """Temporary copy with slot values applied - the stored profile is NOT modified."""
        v = self.values()
        for s in slots:
            if s["subject"] == "self": v[s["attribute"]] = s["value"]
        return v

    # ---- updates
    def apply(self, extraction: dict) -> Dict:
        self.turn += 1
        events = {"set": [], "changed": [], "needs_confirmation": [], "ignored_third_party": [],
                  "hypothetical": extraction["hypothetical"]}
        if extraction["hypothetical"]:                      # never persist what-if statements
            events["whatif_profile"] = self.whatif(extraction["slots"])
            return events
        for s in extraction["slots"]:
            a, v = s["attribute"], s["value"]
            if s["subject"] != "self" and extraction.get("asker") != "relative":
                events["ignored_third_party"].append(s); continue
            old = self.data.get(a)
            if old is None:
                self._write(a, v); events["set"].append((a, v)); continue
            if old["value"] == v: continue
            o = old["value"]
            risky = (not extraction["correction"]) and (
                a in CONFIRM_ALWAYS or (a in BIG_CHANGE and BIG_CHANGE[a](o, v)))
            if risky:
                self.pending[a] = {"value": v, "turn": self.turn}
                events["needs_confirmation"].append({"attribute": a, "old": o, "new": v})
            else:
                self._write(a, v); events["changed"].append((a, o, v))
        return events

    def confirm(self, attribute: str, accept: bool = True):
        p = self.pending.pop(attribute, None)
        if p and accept: self._write(attribute, p["value"])

    def _write(self, attr, value):
        hist = self.data.get(attr, {}).get("history", [])
        if attr in self.data:
            hist = hist + [{"value": self.data[attr]["value"], "turn": self.data[attr]["source_turn"]}]
        self.data[attr] = {"value": value, "source_turn": self.turn, "confidence": 0.9,
                           "updated": time.time(), "history": hist}

    def snapshot(self):
        return copy.deepcopy(self.data)
