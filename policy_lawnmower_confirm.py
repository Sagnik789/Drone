import math


class Policy:
    def __init__(self, info):
        self.a = float(info["arena_size"])
        self.g = int(info["grid_size"])
        self.cell = float(info["cell_size"])
        self.base = tuple(info["base_xy"])
        self.fov = float(info["footprint_tan"])
        self.prior = info["prior_map"]
        self.mean_prior = sum(sum(r) for r in self.prior) / (self.g * self.g) + 1e-12

        # Preserve the successful lawnmower search exactly.
        self.search_h = 40.0
        self.speed = 10.0
        radius = self.search_h * self.fov
        spacing = 1.6 * radius
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

        # Detection handling:
        # keep every candidate, but require an independent second observation
        # for ordinary candidates. This targets isolated false positives while
        # preserving high-prior detections.
        self.pending = []
        self.current = None
        self.done = []
        self.returning = False

        self.last_t = None
        self.max_power = 1.9

    def _prior(self, x, y):
        ix = int(x / self.cell)
        iy = int(y / self.cell)
        if 0 <= ix < self.g and 0 <= iy < self.g:
            return float(self.prior[ix][iy])
        return 0.0

    def _near_done(self, x, y):
        return any(math.hypot(x - dx, y - dy) < 15.0
                   for dx, dy in self.done)

    def _find_pending(self, x, y):
        best = None
        bd = 15.0
        if self.current is not None:
            c = self.current
            d = math.hypot(x - c["x"], y - c["y"])
            if d < bd:
                best, bd = c, d
        for c in self.pending:
            d = math.hypot(x - c["x"], y - c["y"])
            if d < bd:
                best, bd = c, d
        return best

    def _collect(self, obs):
        stamp = float(obs["t"])
        for d in obs.get("detections", []):
            x = float(d["x"])
            y = float(d["y"])

            if self._near_done(x, y):
                continue

            c = self._find_pending(x, y)
            if c is None:
                self.pending.append({
                    "x": x,
                    "y": y,
                    "n": 1,
                    "last_t": stamp,
                    "sumx": x,
                    "sumy": y,
                    "high_prior": self._prior(x, y) >= 2.0 * self.mean_prior,
                })
            elif stamp > c["last_t"] + 1e-6:
                # Only a later simulator observation counts as independent
                # confirmation.
                c["sumx"] += x
                c["sumy"] += y
                c["n"] += 1
                c["x"] = c["sumx"] / c["n"]
                c["y"] = c["sumy"] / c["n"]
                c["last_t"] = stamp

    def _pick_candidate(self, obs):
        if not self.pending:
            return None

        x0 = float(obs["x"])
        y0 = float(obs["y"])

        def key(c):
            prior = min(4.0, self._prior(c["x"], c["y"]) / self.mean_prior)
            distance = math.hypot(c["x"] - x0, c["y"] - y0)
            # Prior is a modest preference; don't reorder the search route.
            return 0.8 * prior + (1.0 if c["n"] >= 2 else 0.0) - distance / 250.0

        return max(self.pending, key=key)

    def _remove(self, c):
        try:
            self.pending.remove(c)
        except ValueError:
            pass

    def _return_cost(self, obs):
        x = float(obs["x"])
        y = float(obs["y"])
        d = math.hypot(x - self.base[0], y - self.base[1])
        return max(d / self.speed, 0.5) * self.max_power * 1.15

    def _declare(self, c):
        x = c["sumx"] / c["n"]
        y = c["sumy"] / c["n"]
        self.done.append((x, y))
        self._remove(c)
        self.current = None
        return {
            "type": "declare",
            "x": x,
            "y": y,
            "triage": "serious",
        }

    def act(self, obs):
        t = float(obs["t"])

        if self.last_t is not None and t > self.last_t:
            p = float(obs.get("energy_used_last_action", 0.0)) / (t - self.last_t)
            if math.isfinite(p) and p > 0:
                self.max_power = max(
                    1.75,
                    min(2.2, 0.995 * self.max_power + 0.005 * p),
                )
        self.last_t = t

        self._collect(obs)

        if not self.returning:
            if float(obs["energy_remaining"]) < self._return_cost(obs) + 48.0:
                self.returning = True

        if self.returning:
            home = math.hypot(
                float(obs["x"]) - self.base[0],
                float(obs["y"]) - self.base[1],
            )
            if home <= 12.0:
                return {"type": "land"}
            return {
                "type": "fly_to",
                "x": self.base[0],
                "y": self.base[1],
                "altitude": self.search_h,
                "speed": self.speed,
            }

        # Finish an outstanding confirmation.
        if self.current is not None:
            c = self.current

            if c["n"] >= 2:
                return self._declare(c)

            # One extra second at 40 m is cheap relative to a false penalty.
            # If no second observation appeared, discard this candidate.
            self._remove(c)
            self.current = None

        # Process candidates generated by the sweep.
        c = self._pick_candidate(obs)
        if c is not None:
            # Strong prior: retain the lawnmower's immediate-declare behavior.
            # This protects genuine high-prior single detections.
            if c["n"] >= 1 and c["high_prior"]:
                return self._declare(c)

            self.current = c
            return {"type": "hover", "duration": 1.0}

        # Continue the exact lawnmower route.
        if self.lane_i >= len(self.lanes):
            self.returning = True
            return {
                "type": "fly_to",
                "x": self.base[0],
                "y": self.base[1],
                "altitude": self.search_h,
                "speed": self.speed,
            }

        x = self.lanes[self.lane_i]
        self.lane_i += 1
        y = self.a if self.up else 0.0
        self.up = not self.up

        return {
            "type": "fly_to",
            "x": x,
            "y": y,
            "altitude": self.search_h,
            "speed": self.speed,
        }
