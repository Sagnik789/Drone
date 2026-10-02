import math


class Policy:
    def __init__(self, info):
        self.a = float(info["arena_size"])
        self.g = int(info["grid_size"])
        self.cell = float(info["cell_size"])
        self.base = tuple(info["base_xy"])
        self.vmax = float(info["speed_range"][1])
        self.climb = float(info["max_climb_rate"])
        self.fov = float(info["footprint_tan"])
        self.prior = info["prior_map"]
        self.mean_prior = sum(sum(row) for row in self.prior) / (self.g * self.g) + 1e-12

        self.search_h = 40.0
        self.verify_h = 12.0
        radius = self.search_h * self.fov
        count = max(2, min(
            16,
            int(math.ceil((self.a - 2.0 * radius) / (1.55 * radius))) + 1,
        ))
        self.lanes = [
            radius + (self.a - 2.0 * radius) * i / (count - 1)
            for i in range(count)
        ]

        values = []
        for x in self.lanes:
            total = 0.0
            for ix in range(self.g):
                if abs((ix + 0.5) * self.cell - x) <= radius:
                    total += sum(self.prior[ix])
            values.append(total)

        remaining = set(range(count))
        self.order = []
        lastx = 0.0
        while remaining:
            j = max(
                remaining,
                key=lambda k: values[k] / (380.0 + abs(self.lanes[k] - lastx)),
            )
            self.order.append(j)
            remaining.remove(j)
            lastx = self.lanes[j]

        self.lane_i = 0
        self.up = True
        self.clusters = []
        self.queue = []
        self.done_xy = []
        self.current = None
        self.stage = 0
        self.returning = False
        self.last_t = None
        self.max_power = 1.7

    def _prior(self, x, y):
        ix, iy = int(x / self.cell), int(y / self.cell)
        if 0 <= ix < self.g and 0 <= iy < self.g:
            return float(self.prior[ix][iy])
        return 0.0

    def _near_done(self, x, y):
        return any(
            math.hypot(x - dx, y - dy) < 10.0
            for dx, dy in self.done_xy
        )

    def _cluster(self, x, y):
        found, best = None, 8.0
        for c in self.clusters:
            if not c["resolved"]:
                d = math.hypot(x - c["x"], y - c["y"])
                if d < best:
                    found, best = c, d
        return found

    def _ingest(self, obs):
        low = float(obs["altitude"]) <= self.verify_h + 1.0

        for d in obs.get("detections", []):
            x = float(d["x"])
            y = float(d["y"])
            tri = d.get("triage")

            if self._near_done(x, y):
                continue

            c = self._cluster(x, y)
            if c is None:
                c = {
                    "x": x,
                    "y": y,
                    "n": 1,
                    "tri": [],
                    "low": [],
                    "resolved": False,
                    "queued": False,
                }
                self.clusters.append(c)
            else:
                n = c["n"]
                c["x"] = (c["x"] * n + x) / (n + 1)
                c["y"] = (c["y"] * n + y) / (n + 1)
                c["n"] = n + 1

            if tri in ("critical", "serious", "minor"):
                c["tri"].append(tri)
                if low:
                    c["low"].append((x, y, tri))

        for c in self.clusters:
            if c["resolved"] or c["queued"]:
                continue
            high_prior_single = (
                c["n"] == 1
                and self._prior(c["x"], c["y"]) >= 2.2 * self.mean_prior
            )
            if c["n"] >= 2 or c["tri"] or high_prior_single:
                c["queued"] = True
                self.queue.append(c)

    def _return_cost(self, obs):
        d = math.hypot(
            float(obs["x"]) - self.base[0],
            float(obs["y"]) - self.base[1],
        )
        duration = max(
            d / self.vmax,
            abs(float(obs["altitude"]) - self.search_h) / self.climb,
            0.5,
        )
        return duration * self.max_power * 1.18

    def _safe(self, obs, c):
        d = math.hypot(
            c["x"] - float(obs["x"]),
            c["y"] - float(obs["y"]),
        )
        go = max(
            d / self.vmax,
            abs(float(obs["altitude"]) - self.verify_h) / self.climb,
            0.5,
        )
        return (
            float(obs["energy_remaining"])
            > self._return_cost(obs) + go * 1.8 + 48.0
        )

    def _pick(self, obs):
        x = float(obs["x"])
        y = float(obs["y"])
        candidates = [c for c in self.queue if not c["resolved"]]
        if not candidates:
            return None

        def score(c):
            confidence = min(c["n"], 4) * 2.0
            confidence += 1.5 if c["tri"] else 0.0
            confidence += min(
                3.0,
                self._prior(c["x"], c["y"]) / self.mean_prior,
            ) * 0.45
            distance = math.hypot(c["x"] - x, c["y"] - y)
            return confidence / (1.0 + distance / 120.0)

        return max(candidates, key=score)

    def _resolve(self, c):
        c["resolved"] = True
        self.current = None
        self.stage = 0

    def act(self, obs):
        t = float(obs["t"])

        if self.last_t is not None and t > self.last_t:
            power = float(obs.get("energy_used_last_action", 0.0)) / (t - self.last_t)
            if power > 0.0 and math.isfinite(power):
                self.max_power = max(self.max_power * 0.996, power)

        self.last_t = t
        self._ingest(obs)

        if (
            not self.returning
            and float(obs["energy_remaining"]) < self._return_cost(obs) + 52.0
        ):
            self.returning = True

        if self.returning:
            home_dist = math.hypot(
                float(obs["x"]) - self.base[0],
                float(obs["y"]) - self.base[1],
            )
            if home_dist <= 12.0:
                return {"type": "land"}

            return {
                "type": "fly_to",
                "x": self.base[0],
                "y": self.base[1],
                "altitude": self.search_h,
                "speed": self.vmax,
            }

        if self.current is not None:
            c = self.current

            if c["low"]:
                votes = {"critical": 0, "serious": 0, "minor": 0}
                for _, _, tri in c["low"]:
                    votes[tri] += 1

                tri = max(votes, key=votes.get)
                points = [
                    (x, y)
                    for x, y, label in c["low"]
                    if label == tri
                ]
                x = sum(point[0] for point in points) / len(points)
                y = sum(point[1] for point in points) / len(points)

                self.done_xy.append((x, y))
                self._resolve(c)
                return {
                    "type": "declare",
                    "x": x,
                    "y": y,
                    "triage": tri,
                }

            if self.stage < 2:
                self.stage += 1
                return {"type": "hover", "duration": 1.5}

            self._resolve(c)

        c = self._pick(obs)
        if c is not None:
            d = math.hypot(
                c["x"] - float(obs["x"]),
                c["y"] - float(obs["y"]),
            )
            if d <= 115.0 and self._safe(obs, c):
                self.current = c
                self.stage = 0
                return {
                    "type": "fly_to",
                    "x": c["x"],
                    "y": c["y"],
                    "altitude": self.verify_h,
                    "speed": self.vmax,
                }

        if self.lane_i >= len(self.order):
            if c is not None and self._safe(obs, c):
                self.current = c
                self.stage = 0
                return {
                    "type": "fly_to",
                    "x": c["x"],
                    "y": c["y"],
                    "altitude": self.verify_h,
                    "speed": self.vmax,
                }

            self.returning = True
            return {
                "type": "fly_to",
                "x": self.base[0],
                "y": self.base[1],
                "altitude": self.search_h,
                "speed": self.vmax,
            }

        lane = self.order[self.lane_i]
        self.lane_i += 1
        y = self.a if self.up else 0.0
        self.up = not self.up

        return {
            "type": "fly_to",
            "x": self.lanes[lane],
            "y": y,
            "altitude": self.search_h,
            "speed": self.vmax,
        }