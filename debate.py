from dataclasses import dataclass, field
from time import time


@dataclass
class Claim:
    text: str
    result: dict
    timestamp: float = field(default_factory=time)


@dataclass
class Side:
    name: str
    claims: list[Claim] = field(default_factory=list)

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.claims if c.result["status"] == "supported")

    @property
    def refuted_count(self) -> int:
        return sum(1 for c in self.claims if c.result["status"] == "refuted")

    @property
    def unverifiable_count(self) -> int:
        return sum(1 for c in self.claims if c.result["status"] == "unverifiable")

    @property
    def accuracy(self) -> float:
        verifiable = self.supported_count + self.refuted_count
        if verifiable == 0:
            return 0.0
        return self.supported_count / verifiable

    def summary(self) -> dict:
        return {
            "name": self.name,
            "total_claims": len(self.claims),
            "supported": self.supported_count,
            "refuted": self.refuted_count,
            "unverifiable": self.unverifiable_count,
            "accuracy": round(self.accuracy * 100, 1),
        }


class Debate:
    def __init__(self, side_a_name: str = "Side A", side_b_name: str = "Side B"):
        self.sides = {
            "a": Side(name=side_a_name),
            "b": Side(name=side_b_name),
        }
        self.active_side: str | None = None

    def set_active_side(self, side: str):
        self.active_side = side.lower()

    def add_claim(self, text: str, result: dict, side: str | None = None):
        side_key = (side or self.active_side or "a").lower()
        self.sides[side_key].claims.append(Claim(text=text, result=result))

    def get_side_summary(self, side: str) -> dict:
        return self.sides[side.lower()].summary()

    def get_side_claims(self, side: str) -> list[dict]:
        return [
            {"text": c.text, "result": c.result}
            for c in self.sides[side.lower()].claims
        ]

    def get_full_summary(self) -> dict:
        summaries = {k: s.summary() for k, s in self.sides.items()}
        # Determine winner by accuracy, then by supported count
        a, b = self.sides["a"], self.sides["b"]
        if a.accuracy > b.accuracy:
            winner = "a"
        elif b.accuracy > a.accuracy:
            winner = "b"
        elif a.supported_count > b.supported_count:
            winner = "a"
        elif b.supported_count > a.supported_count:
            winner = "b"
        else:
            winner = "tie"

        return {
            "sides": summaries,
            "winner": winner,
            "winner_name": self.sides[winner].name if winner != "tie" else "Tie",
        }

    def reset(self):
        for s in self.sides.values():
            s.claims.clear()
