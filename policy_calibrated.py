import math


class Policy:
    def __init__(self, info):
        self.arena = float(info["arena_size"])
        self.grid = int(info["grid_size"])
        self.cell = float(info["cell_size"])
        self.base = tuple(info["base_xy"])
        self.budget = float(info["energy_budget"])
        self.vmax = float(info["speed_range"][1])
        self.climb = float(info["max_climb_rate"])
        self.fov = float(info["footprint_tan"])
        self.prior = info["prior_map"]

        # Empirically supported operating points from the calibration data:
        # 30 m gives better detection/position accuracy than 40 m at lower
        # altitude power, while 10 m/s is a tested, moderate-speed point.
        self.search_h = min(30.0, float(info["altitude_range"][1]))
        self.search_v = min(10.0, self.vmax)

        # Triage was observed only at <=10 m, so verification must actually
        # descend into that regime.
        self.verify_h = min(10.0, float(info["altitude_range"][1]))
        self.verify_v = min(10.0, self.vmax)

        # At 30 m the footprint radius is ~25.2 m.  Use overlapping lanes
        # rather than the wider 40 m spacing of the previous policy.
        radius = self.search_h * self.fov
        spacing = 1.50 * radius
        n = max(2, int(math.ceil((self.arena - 2.0 * radius) / spacing)) + 1)
        n = min(n, 16)
        if n == 1:
            self.lanes = [self.arena * 0.5]
        else:
            self.lanes = [
                radius + (self.arena - 2.0 * radius) * i / (n - 1)
                for i in range(n)
            ]

        # Prior-guided lane ordering, but all lanes remain covered.
        centers = [(i + 0.5) * self.cell for i in range(self.grid)]
        lane_value = []
        for x in self.lanes:
            s = 0.0
            for ix, cx in enumerate(centers):
                if abs(cx - x) <= radius:
                    s += sum(float(v) for v in self.prior[ix])
            lane_value.append(s)

        remaining = set(range(len(self.lanes)))
        self.order = []
        last_x = self.base[0]
        while remaining:
            j = max(
                remaining,
                key=lambda k: lane_value[k] / (self.arena + abs(self.lanes[k] - last_x))
            )
            self.order.append(j)
            remaining.remove(j)
            last_x = self.lanes[j]

        self.lane_i = 0
        self.up = True

        self.clusters = []
        self.queue = []
        self.current = None
        self.stage = 0
        self.returning = False
        self.last_t = None
        self.power_est = 1.8
        self.max_power = 2.2

        self.declared_xy = []

    def _dist(self, x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    def _prior(self, x, y):
        ix = int(x / self.cell)
        iy = int(y / self.cell)
        if 0 <= ix < self.grid and 0 <= iy < self.grid:
            return float(self.prior[ix][iy])
        return 0.0

    def _near_declared(self, x, y, r=10.0):
        return any(self._dist(x, y, dx, dy) < r for dx, dy in self.declared_xy)

    def _find_cluster(self, x, y):
        best = None
        best_d = 7.0
        for c in self.clusters:
            if c["resolved"]:
                continue
            d = self._dist(x, y, c["x"], c["y"])
            if d < best_d:
                best_d = d
                best = c
        return best

    def _ingest(self, detections):
        for d in detections:
            x = float(d["x"])
            y = float(d["y"])
            if self._near_declared(x, y):
                continue

            tri = d.get("triage")
            c = self._find_cluster(x, y)

            if c is None:
                c = {
                    "x": x,
                    "y": y,
                    "hits": 1,
                    "tri": [],
                    "resolved": False,
                    "queued": False,
                }
                self.clusters.append(c)
            else:
                n = c["hits"]
                c["x"] = (c["x"] * n + x) / (n + 1.0)
                c["y"] = (c["y"] * n + y) / (n + 1.0)
                c["hits"] = n + 1

            if tri in ("critical", "serious", "minor"):
                c["tri"].append(tri)

        mean_prior = sum(sum(float(v) for v in row) for row in self.prior)
        mean_prior /= max(1, self.grid * self.grid)
        mean_prior += 1e-12

        for c in self.clusters:
            if c["resolved"] or c["queued"]:
                continue

            ratio = self._prior(c["x"], c["y"]) / mean_prior

            # A repeated detection is the main high-altitude candidate signal.
            # A single hit is queued only when prior support is strong.
            if c["hits"] >= 2 or c["tri"] or ratio >= 2.0:
                c["queued"] = True
                self.queue.append(c)

    def _triage(self, c):
        if not c["tri"]:
            return None
        counts = {"critical": 0, "serious": 0, "minor": 0}
        for t in c["tri"]:
            counts[t] += 1
        return max(counts, key=counts.get)

    def _return_cost(self, obs):
        d = self._dist(
            float(obs["x"]), float(obs["y"]),
            self.base[0], self.base[1]
        )
        h = float(obs["altitude"])
        duration = max(
            d / max(self.vmax, 1.0),
            abs(h - self.search_h) / self.climb,
            0.5
        )
        return duration * max(2.0, self.max_power * 1.15)

    def _safe(self, obs, c):
        d = self._dist(float(obs["x"]), float(obs["y"]), c["x"], c["y"])
        duration = max(
            d / max(self.verify_v, 1.0),
            abs(float(obs["altitude"]) - self.verify_h) / self.climb,
            0.5
        )

        # Budget for reaching the candidate, a short verification hover,
        # declaration, and a conservative return reserve.
        verification_cost = duration * 2.0 + 4.0
        return float(obs["energy_remaining"]) > (
            self._return_cost(obs) + verification_cost + 55.0
        )

    def _choose_candidate(self, obs):
        if not self.queue:
            return None

        x = float(obs["x"])
        y = float(obs["y"])

        best = None
        best_score = -1e30
        for c in self.queue:
            if c["resolved"]:
                continue

            d = self._dist(c["x"], c["y"], x, y)
            p = self._prior(c["x"], c["y"])
            confidence = 2.0 * min(c["hits"], 5)
            confidence += 2.5 if c["tri"] else 0.0
            confidence += min(3.0, p / (
                sum(sum(float(v) for v in row) for row in self.prior)
                / max(1, self.grid * self.grid) + 1e-12
            )) * 0.5

            # Nearby candidates are cheaper to verify, but confidence remains
            # more important than distance.
            score = confidence / (1.0 + d / 120.0)
            if score > best_score:
                best_score = score
                best = c

        return best

    def _finish_candidate(self, c):
        c["resolved"] = True
        self.queue = [q for q in self.queue if q is not c]
        self.current = None
        self.stage = 0

    def act(self, obs):
        t = float(obs["t"])

        if self.last_t is not None:
            dt = t - self.last_t
            if dt > 1e-9:
                used = float(obs.get("energy_used_last_action", 0.0))
                p = used / dt
                if p > 0.0 and math.isfinite(p):
                    self.power_est = 0.8 * self.power_est + 0.2 * p
                    self.max_power = max(self.max_power * 0.995, p)

        self.last_t = t
        self._ingest(obs.get("detections", []))

        # Keep a large reserve because the hidden wind can change the actual
        # return cost.  Never continue searching once the reserve is reached.
        if not self.returning:
            if float(obs["energy_remaining"]) <= self._return_cost(obs) + 55.0:
                self.returning = True

        if self.returning:
            d = self._dist(
                float(obs["x"]), float(obs["y"]),
                self.base[0], self.base[1]
            )
            if d <= 12.0:
                return {"type": "land"}

            return {
                "type": "fly_to",
                "x": self.base[0],
                "y": self.base[1],
                "altitude": self.search_h,
                "speed": self.search_v,
            }

        # Candidate verification: descend to 10 m, then use one or two short
        # observation intervals to obtain reliable position + triage.
        if self.current is not None:
            c = self.current

            tri = self._triage(c)
            if tri is not None:
                x, y = c["x"], c["y"]
                self.declared_xy.append((x, y))
                self._finish_candidate(c)
                return {
                    "type": "declare",
                    "x": x,
                    "y": y,
                    "triage": tri,
                }

            if self.stage == 0:
                self.stage = 1
                return {"type": "hover", "duration": 1.0}

            if self.stage == 1:
                self.stage = 2
                return {"type": "hover", "duration": 1.0}

            # No low-altitude detection after two seconds: do not blindly
            # declare a high-altitude noisy coordinate.
            self._finish_candidate(c)

        candidate = self._choose_candidate(obs)

        if candidate is not None:
            d = self._dist(
                float(obs["x"]), float(obs["y"]),
                candidate["x"], candidate["y"]
            )
            if d <= 120.0 and self._safe(obs, candidate):
                self.current = candidate
                self.stage = 0
                return {
                    "type": "fly_to",
                    "x": candidate["x"],
                    "y": candidate["y"],
                    "altitude": self.verify_h,
                    "speed": self.verify_v,
                }

        # Complete deterministic coverage sweep.
        if self.lane_i < len(self.order):
            lane = self.order[self.lane_i]
            self.lane_i += 1

            y = self.arena if self.up else 0.0
            self.up = not self.up

            return {
                "type": "fly_to",
                "x": self.lanes[lane],
                "y": y,
                "altitude": self.search_h,
                "speed": self.search_v,
            }

        # After coverage, verify any remaining strong candidates if safe.
        candidate = self._choose_candidate(obs)
        if candidate is not None and self._safe(obs, candidate):
            self.current = candidate
            self.stage = 0
            return {
                "type": "fly_to",
                "x": candidate["x"],
                "y": candidate["y"],
                "altitude": self.verify_h,
                "speed": self.verify_v,
            }

        self.returning = True
        return {
            "type": "fly_to",
            "x": self.base[0],
            "y": self.base[1],
            "altitude": self.search_h,
            "speed": self.search_v,
        }
