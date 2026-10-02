import math


class Policy:
    def __init__(self, info):
        self.a = float(info["arena_size"])
        self.fov = float(info["footprint_tan"])
        self.base = tuple(info["base_xy"])
        self.vmax = float(info["speed_range"][1])
        self.prior = info["prior_map"]
        self.g = int(info["grid_size"])
        self.cell = float(info["cell_size"])
        self.budget = float(info["energy_budget"])

        # The calibrated data plus the supplied lawnmower baseline point to
        # 40 m / 10 m/s as the useful coverage regime.  Lower altitude needs
        # too many lanes for the available battery.
        self.h = min(40.0, float(info["altitude_range"][1]))
        self.v = min(10.0, self.vmax)

        radius = self.h * self.fov
        spacing = 1.60 * radius

        # Same basic geometry as the successful lawnmower strategy.
        n = max(2, int(math.ceil((self.a - 2.0 * radius) / spacing)) + 1)
        self.lanes = [
            radius + (self.a - 2.0 * radius) * i / (n - 1)
            for i in range(n)
        ]

        # Keep lanes spatially ordered. Reordering them by prior value can
        # create expensive cross-arena jumps and reduce actual coverage.
        self.lane_i = 0
        self.up = True

        # High-altitude detections are noisy, so deduplicate aggressively.
        # At 40 m the calibration position-error distribution is still much
        # better than at 50-80 m, while false positives were rare in the
        # calibration experiment.
        self.dedup_r = 15.0
        self.declared = []

        self.returning = False
        self.last_t = None
        self.power = 2.2

        # Candidates seen during a flight action. We declare at the next
        # decision point rather than issuing multiple declarations from one
        # observation, which keeps action count and duplicate reports down.
        self.pending = []

    def _d(self, x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    def _near(self, x, y, pts, r):
        for px, py in pts:
            if self._d(x, y, px, py) < r:
                return True
        return False

    def _return_cost(self, obs):
        x = float(obs["x"])
        y = float(obs["y"])
        d = self._d(x, y, self.base[0], self.base[1])

        # Conservative but not so large that it prevents useful coverage.
        # Use the empirical maximum observed operating power with margin.
        return (d / self.v) * 2.5 + 20.0

    def _ingest(self, obs):
        for det in obs.get("detections", []):
            x = float(det["x"])
            y = float(det["y"])

            if self._near(x, y, self.declared, self.dedup_r):
                continue
            if self._near(x, y, self.pending, self.dedup_r):
                continue

            self.pending.append((x, y))

    def _best_pending(self, obs):
        if not self.pending:
            return None

        x = float(obs["x"])
        y = float(obs["y"])

        # Prefer nearby detections so declarations do not create large
        # excursions away from the coverage sweep.
        best_i = min(
            range(len(self.pending)),
            key=lambda i: self._d(x, y, self.pending[i][0], self.pending[i][1])
        )
        return best_i

    def act(self, obs):
        t = float(obs["t"])

        if self.last_t is not None:
            dt = t - self.last_t
            if dt > 1e-9:
                used = float(obs.get("energy_used_last_action", 0.0))
                p = used / dt
                if p > 0.0 and math.isfinite(p):
                    self.power = 0.8 * self.power + 0.2 * p

        self.last_t = t
        self._ingest(obs)

        e = float(obs["energy_remaining"])

        # Preserve enough energy to return and land. This is intentionally
        # conservative because wind varies by scenario.
        if not self.returning and e <= self._return_cost(obs):
            self.returning = True

        if self.returning:
            x = float(obs["x"])
            y = float(obs["y"])
            if self._d(x, y, self.base[0], self.base[1]) <= 12.0:
                return {"type": "land"}

            return {
                "type": "fly_to",
                "x": self.base[0],
                "y": self.base[1],
                "altitude": self.h,
                "speed": self.v,
            }

        # Declare detections at the coverage altitude. This deliberately
        # follows the successful lawnmower baseline: descending every
        # candidate costs too much energy and would sharply reduce coverage.
        i = self._best_pending(obs)
        if i is not None:
            x, y = self.pending.pop(i)

            self.declared.append((x, y))
            return {
                "type": "declare",
                "x": x,
                "y": y,
                "triage": "serious",
            }

        if self.lane_i < len(self.lanes):
            x = self.lanes[self.lane_i]
            y = self.a if self.up else 0.0
            self.lane_i += 1
            self.up = not self.up

            return {
                "type": "fly_to",
                "x": x,
                "y": y,
                "altitude": self.h,
                "speed": self.v,
            }

        self.returning = True
        return {
            "type": "fly_to",
            "x": self.base[0],
            "y": self.base[1],
            "altitude": self.h,
            "speed": self.v,
        }
