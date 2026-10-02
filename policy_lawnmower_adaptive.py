import math


class Policy:
    def __init__(self, info):
        self.a = float(info["arena_size"])
        self.g = int(info["grid_size"])
        self.cell = float(info["cell_size"])
        self.base = tuple(info["base_xy"])
        self.fov = float(info["footprint_tan"])
        self.prior = info["prior_map"]
        self.vmax = float(info["speed_range"][1])
        self.budget = float(info["energy_budget"])

        # Preserve the proven lawnmower search geometry.
        self.h = min(40.0, float(info["altitude_range"][1]))
        self.v = min(10.0, self.vmax)
        radius = self.h * self.fov
        spacing = 1.60 * radius
        count = max(
            2,
            min(16, int(math.ceil((self.a - 2.0 * radius) / spacing)) + 1),
        )
        self.lanes = [
            radius + (self.a - 2.0 * radius) * i / max(1, count - 1)
            for i in range(count)
        ]
        self.lane_i = 0
        self.up = True

        # The prior is used only as a light-touch false-positive filter.
        # Most detections retain the lawnmower's immediate declaration.
        vals = []
        for row in self.prior:
            for z in row:
                try:
                    z = float(z)
                    if math.isfinite(z):
                        vals.append(z)
                except Exception:
                    pass
        vals.sort()
        self.mean_prior = (sum(vals) / len(vals)) if vals else 0.0
        self.low_prior_cut = vals[max(0, int(0.15 * (len(vals) - 1)))] if vals else 0.0
        self.high_prior_cut = vals[max(0, int(0.75 * (len(vals) - 1)))] if vals else 0.0

        self.dedup_r = 15.0
        self.declared = []
        self.pending = []
        self.returning = False
        self.last_t = None
        self.power = 1.9

    def _dist(self, x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    def _prior(self, x, y):
        ix = int(x / self.cell)
        iy = int(y / self.cell)
        if 0 <= ix < self.g and 0 <= iy < self.g:
            try:
                return float(self.prior[ix][iy])
            except Exception:
                return 0.0
        return 0.0

    def _near(self, x, y, pts, r):
        return any(self._dist(x, y, px, py) < r for px, py in pts)

    def _find_candidate(self, x, y):
        best = None
        bd = self.dedup_r
        for c in self.pending:
            d = self._dist(x, y, c["x"], c["y"])
            if d < bd:
                best = c
                bd = d
        return best

    def _ingest(self, obs):
        stamp = float(obs["t"])
        for det in obs.get("detections", []):
            x = float(det["x"])
            y = float(det["y"])

            if self._near(x, y, self.declared, self.dedup_r):
                continue

            c = self._find_candidate(x, y)
            if c is None:
                p = self._prior(x, y)
                self.pending.append({
                    "x": x,
                    "y": y,
                    "sumx": x,
                    "sumy": y,
                    "n": 1,
                    "last_t": stamp,
                    "prior": p,
                    "seen_actions": 0,
                })
            elif stamp > c["last_t"] + 1e-6:
                c["sumx"] += x
                c["sumy"] += y
                c["n"] += 1
                c["x"] = c["sumx"] / c["n"]
                c["y"] = c["sumy"] / c["n"]
                c["last_t"] = stamp

    def _mark_action(self):
        for c in self.pending:
            c["seen_actions"] += 1

    def _pick(self):
        if not self.pending:
            return None

        # Prefer confirmed candidates, then higher-prior candidates.
        return max(
            self.pending,
            key=lambda c: (
                1 if c["n"] >= 2 else 0,
                min(c["prior"], self.high_prior_cut),
                -c["seen_actions"],
            ),
        )

    def _remove(self, c):
        try:
            self.pending.remove(c)
        except ValueError:
            pass

    def _declare(self, c):
        x = c["sumx"] / c["n"]
        y = c["sumy"] / c["n"]
        self._remove(c)
        self.declared.append((x, y))
        return {
            "type": "declare",
            "x": x,
            "y": y,
            "triage": "serious",
        }

    def _return_cost(self, obs):
        x = float(obs["x"])
        y = float(obs["y"])
        d = self._dist(x, y, self.base[0], self.base[1])
        # Conservative enough for the calibrated wind variation, but keep
        # essentially all of the lawnmower's usable search time.
        return (d / self.v) * 2.25 + 18.0

    def act(self, obs):
        t = float(obs["t"])

        if self.last_t is not None and t > self.last_t:
            p = float(obs.get("energy_used_last_action", 0.0)) / (t - self.last_t)
            if p > 0.0 and math.isfinite(p):
                self.power = 0.9 * self.power + 0.1 * p
        self.last_t = t

        self._ingest(obs)

        if not self.returning and float(obs["energy_remaining"]) <= self._return_cost(obs):
            self.returning = True

        if self.returning:
            x = float(obs["x"])
            y = float(obs["y"])
            if self._dist(x, y, self.base[0], self.base[1]) <= 12.0:
                return {"type": "land"}
            return {
                "type": "fly_to",
                "x": self.base[0],
                "y": self.base[1],
                "altitude": self.h,
                "speed": self.v,
            }

        self._mark_action()

        c = self._pick()
        if c is not None:
            # Confirmed detections are always accepted.
            if c["n"] >= 2:
                return self._declare(c)

            # High-prior single detections retain the baseline behavior.
            if c["prior"] >= self.high_prior_cut:
                return self._declare(c)

            # Only the very lowest-prior tail is filtered. If it does not get
            # a natural confirmation by the next decision, discard it rather
            # than spending a hover or diverting from the sweep.
            if c["prior"] <= self.low_prior_cut:
                if c["seen_actions"] >= 2:
                    self._remove(c)
                else:
                    # Give the normal sweep one more action to generate a
                    # natural confirmation, without any extra hover.
                    pass
            else:
                # Middle-prior detections retain immediate-declare behavior;
                # this protects recall and preserves the baseline's value.
                return self._declare(c)

        # Continue exactly along the lawnmower route.
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

        # No more coverage. Give remaining candidates one final chance before
        # returning; only confirmed or non-low-prior candidates are declared.
        c = self._pick()
        if c is not None:
            if c["n"] >= 2 or c["prior"] > self.low_prior_cut:
                return self._declare(c)
            self._remove(c)

        self.returning = True
        return {
            "type": "fly_to",
            "x": self.base[0],
            "y": self.base[1],
            "altitude": self.h,
            "speed": self.v,
        }
