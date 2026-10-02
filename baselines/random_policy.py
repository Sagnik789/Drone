# Baseline: random actions, never returns to base.
import random


class Policy:
    def __init__(self, info):
        self.rng = random.Random(12345)
        self.arena = info["arena_size"]
        self.alt = info["altitude_range"]
        self.speed = info["speed_range"]

    def act(self, obs):
        r = self.rng.random()
        if r < 0.05:
            return {"type": "declare", "x": self.rng.uniform(0, self.arena),
                    "y": self.rng.uniform(0, self.arena),
                    "triage": self.rng.choice(["critical", "serious", "minor"])}
        if r < 0.20:
            return {"type": "hover", "duration": self.rng.uniform(0.5, 10.0)}
        return {"type": "fly_to", "x": self.rng.uniform(0, self.arena), "y": self.rng.uniform(0, self.arena),
                "altitude": self.rng.uniform(*self.alt), "speed": self.rng.uniform(*self.speed)}
