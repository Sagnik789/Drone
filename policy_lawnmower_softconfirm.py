import math


class Policy:
    def __init__(self, info):
        self.a = float(info["arena_size"])
        self.fov = float(info["footprint_tan"])
        self.base = tuple(info["base_xy"])
        self.v = min(10.0, float(info["speed_range"][1]))
        self.h = min(40.0, float(info["altitude_range"][1]))
        self.prior = info["prior_map"]
        self.g = int(info["grid_size"])
        self.cell = float(info["cell_size"])

        # Keep the proven lawnmower geometry unchanged.
        radius = self.h * self.fov
        spacing = 1.60 * radius
        n = max(2, int(math.ceil((self.a - 2.0 * radius) / spacing)) + 1)
        self.lanes = [
            radius + (self.a - 2.0 * radius) * i / (n - 1)
            for i in range(n)
        ]

        self.lane_i = 0
        self.up = True

        # Keep the baseline's 15 m spatial deduplication.
        self.dedup_r = 15.0

        self.declared = []

        # Candidate format:
        # [x, y, support_count, last_seen_t]
        self.pending = []

        # Mean prior is only used for deciding which isolated detections
        # deserve a little more evidence. It does not alter the flight path.
        total = 0.0
        for row in self.prior:
            total += sum(float(v) for v in row)
        self.mean_prior = total / max(1, self.g * self.g)

        self.returning = False
        self.last_t = None
        self.power = 2.2

    @staticmethod
    def _dist(x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    def _near_points(self, x, y, points, r):
        for px, py in points:
            if self._dist(x, y, px, py) <= r:
                return True
        return False

    def _prior_at(self, x, y):
        ix = int(x / self.cell)
        iy = int(y / self.cell)
        if 0 <= ix < self.g and 0 <= iy < self.g:
            return float(self.prior[ix][iy])
        return 0.0

    def _find_pending(self, x, y):
        best = None
        best_d = self.dedup_r
        for i, c in enumerate(self.pending):
            d = self._dist(x, y, c[0], c[1])
            if d <= best_d:
                best = i
                best_d = d
        return best

    def _ingest(self, obs):
        t = float(obs["t"])

        for det in obs.get("detections", []):
            x = float(det["x"])
            y = float(det["y"])

            if self._near_points(x, y, self.declared, self.dedup_r):
                continue

            i = self._find_pending(x, y)
            if i is None:
                self.pending.append([x, y, 1, t])
            else:
                c = self.pending[i]
                n = c[2]
                # Average repeated measurements instead of using the first
                # noisy coordinate.
                c[0] = (c[0] * n + x) / (n + 1)
                c[1] = (c[1] * n + y) / (n + 1)
                c[2] = n + 1
                c[3] = t

    def _candidate_priority(self, c):
        return self._prior_at(c[0], c[1]) / max(self.mean_prior, 1e-12)

    def _choose_declaration(self, obs, force=False):
        if not self.pending:
            return None

        x = float(obs["x"])
        y = float(obs["y"])

        # Repeated natural detections are strong evidence. They are declared
        # immediately without changing the flight path.
        repeated = [
            (i, c) for i, c in enumerate(self.pending) if c[2] >= 2
        ]
        if repeated:
            i, c = min(
                repeated,
                key=lambda z: self._dist(x, y, z[1][0], z[1][1])
            )
            return i

        # Preserve the baseline's recall for normal/high-prior detections.
        # Only the very lowest-prior isolated detections are held.
        immediate = []
        for i, c in enumerate(self.pending):
            ratio = self._candidate_priority(c)
            if ratio >= 0.35:
                immediate.append((i, c))

        if immediate:
            i, c = min(
                immediate,
                key=lambda z: self._dist(x, y, z[1][0], z[1][1])
            )
            return i

        # A held isolated candidate is only declared at the end of the
        # sweep. This avoids sacrificing genuine one-shot detections forever.
        if force:
            return min(
                range(len(self.pending)),
                key=lambda i: self._dist(
                    x, y, self.pending[i][0], self.pending[i][1]
                )
            )

        return None

    def _return_cost(self, obs):
        x = float(obs["x"])
        y = float(obs["y"])
        d = self._dist(x, y, self.base[0], self.base[1])
        return (d / self.v) * 2.5 + 20.0

    def _declare_pending(self, i):
        c = self.pending.pop(i)
        x, y = c[0], c[1]
        self.declared.append((x, y))
        return {
            "type": "declare",
            "x": x,
            "y": y,
            "triage": "serious",
        }

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
        if not self.returning and e <= self._return_cost(obs):
            self.returning = True

        if self.returning:
            x = float(obs["x"])
            y = float(obs["y"])

            # Before committing to the return, use any remaining candidate
            # only if it is already a naturally repeated detection.
            i = self._choose_declaration(obs, force=False)
            if i is not None:
                return self._declare_pending(i)

            if self._dist(x, y, self.base[0], self.base[1]) <= 12.0:
                return {"type": "land"}

            return {
                "type": "fly_to",
                "x": self.base[0],
                "y": self.base[1],
                "altitude": self.h,
                "speed": self.v,
            }

        # Normal operation: declare high-confidence/repeated detections
        # without changing the proven lawnmower trajectory.
        i = self._choose_declaration(obs, force=False)
        if i is not None:
            return self._declare_pending(i)

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

        # Sweep complete. Do not discard isolated low-prior candidates.
        i = self._choose_declaration(obs, force=True)
        if i is not None:
            return self._declare_pending(i)

        self.returning = True
        return {
            "type": "fly_to",
            "x": self.base[0],
            "y": self.base[1],
            "altitude": self.h,
            "speed": self.v,
        }
