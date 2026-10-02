# Baseline: lane sweep at 40 m, declares every new detection as "serious",
# flies home when energy drops below a simple margin.
import math

ALT = 40.0
SPEED = 10.0
DEDUP_RADIUS = 15.0
SEGMENT = 50.0


class Policy:
    def __init__(self, info):
        self.arena = info["arena_size"]
        self.budget = info["energy_budget"]
        r = ALT * info.get("footprint_tan", math.tan(math.radians(40.0)))
        spacing = 1.6 * r
        self.waypoints = []
        x, up = spacing / 2.0, True
        while x < self.arena:
            ys = [0.0, self.arena] if up else [self.arena, 0.0]
            n = int(math.ceil(self.arena / SEGMENT))
            for i in range(n + 1):
                self.waypoints.append((x, ys[0] + (ys[1] - ys[0]) * i / n))
            x += spacing
            up = not up
        self.wp = 0
        self.declared = []
        self.pending = []
        self.returning = False
        self.power_est = 2.0
        self.last = None

    def act(self, obs):
        x, y = obs["x"], obs["y"]
        if self.last is not None and self.last[0] == "fly_to" and obs["t"] > self.last[1]:
            dt = obs["t"] - self.last[1]
            if dt > 0:
                self.power_est = 0.7 * self.power_est + 0.3 * obs["energy_used_last_action"] / dt

        for d in obs["detections"]:
            if all(math.hypot(d["x"] - px, d["y"] - py) > DEDUP_RADIUS for px, py in self.declared):
                self.declared.append((d["x"], d["y"]))
                self.pending.append((d["x"], d["y"]))
        if self.pending:
            px, py = self.pending.pop(0)
            self.last = ("declare", obs["t"])
            return {"type": "declare", "x": px, "y": py, "triage": "serious"}

        home_dist = math.hypot(x, y)
        if not self.returning:
            ret_cost = home_dist / SPEED * self.power_est * 1.5 + 0.05 * self.budget
            if obs["energy_remaining"] < ret_cost or self.wp >= len(self.waypoints):
                self.returning = True
        if self.returning:
            if home_dist <= 10.0:
                return {"type": "land"}
            self.last = ("fly_to", obs["t"])
            return {"type": "fly_to", "x": 0.0, "y": 0.0, "altitude": ALT, "speed": SPEED}

        tx, ty = self.waypoints[self.wp]
        self.wp += 1
        self.last = ("fly_to", obs["t"])
        return {"type": "fly_to", "x": tx, "y": ty, "altitude": ALT, "speed": SPEED}
