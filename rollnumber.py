
import math
import numpy as np


class Policy:
    def __init__(self, info):
        self.arena = float(info["arena_size"])
        self.cell = float(info["cell_size"])
        self.grid = int(info["grid_size"])
        self.base = tuple(info["base_xy"])
        self.budget = float(info["energy_budget"])
        self.alt_min, self.alt_max = map(float, info["altitude_range"])
        self.vmin, self.vmax = map(float, info["speed_range"])
        self.max_climb = float(info["max_climb_rate"])
        self.tan_fov = float(info["footprint_tan"])
        self.match_r = float(info["match_radius"])
        self.triage_levels = ("critical", "serious", "minor")

        self.search_alt = 30.0
        self.search_speed = self.vmax
        self.verify_alt = min(14.0, self.alt_max)
        self.verify_speed = self.vmax

        self.prior = np.asarray(info["prior_map"], dtype=float)
        if self.prior.shape != (self.grid, self.grid):
            self.prior = self.prior.reshape((self.grid, self.grid))
        self.prior = np.maximum(self.prior, 0.0)
        self.prior_mean = float(np.mean(self.prior)) + 1e-12
        self.prior_max = float(np.max(self.prior)) + 1e-12

        # Fixed deterministic coverage lanes.  Spacing is chosen from the
        # actual sensor footprint, with enough overlap to avoid thin gaps.
        r = self.search_alt * self.tan_fov
        n = max(2, int(math.ceil((self.arena - 2.0 * r) / (1.55 * r))) + 1)
        n = min(n, 16)
        if n == 1:
            self.lanes = [self.arena * 0.5]
        else:
            self.lanes = [
                r + (self.arena - 2.0 * r) * i / (n - 1)
                for i in range(n)
            ]
        self.lanes = [min(max(x, 0.0), self.arena) for x in self.lanes]

        # Lane value = prior mass in the strip seen by that lane.
        centers_x = (np.arange(self.grid) + 0.5) * self.cell
        lane_value = []
        for x in self.lanes:
            mask = np.abs(centers_x - x) <= r
            lane_value.append(float(np.sum(self.prior[mask, :])))
        self.lane_value = lane_value

        # Choose a high-value-first order while retaining deterministic routing.
        # The travel between successive full-height lanes is cheap relative to
        # the 400 m sweep itself, so prior mass is the dominant criterion.
        remaining = set(range(len(self.lanes)))
        order = []
        px = 0.0
        while remaining:
            best = None
            best_score = -1.0
            for j in remaining:
                dx = abs(self.lanes[j] - px)
                score = self.lane_value[j] / (380.0 + dx)
                if score > best_score:
                    best_score = score
                    best = j
            order.append(best)
            px = self.lanes[best]
            remaining.remove(best)
        self.lane_order = order
        self.lane_i = 0
        self.go_up = True

        self.clusters = []
        self.declared = []
        self.verify_queue = []
        self.current_verify = None
        self.verify_stage = 0
        self.returning = False

        self.last_t = None
        self.power_est = 1.25
        self.max_power_est = 1.6
        self.search_done = False
        self._seen_action = 0

    def _dist(self, x, y, x2=0.0, y2=0.0):
        return math.hypot(x - x2, y - y2)

    def _cell_prior(self, x, y):
        ix = int(x / self.cell)
        iy = int(y / self.cell)
        if ix < 0 or iy < 0 or ix >= self.grid or iy >= self.grid:
            return 0.0
        return float(self.prior[ix, iy])

    def _candidate_score(self, c):
        # Multiple independent-looking detections are much stronger than one.
        # Prior contributes, but is deliberately capped so the prior cannot
        # suppress genuine low-prior survivors.
        p = self._cell_prior(c["x"], c["y"]) / self.prior_mean
        p = min(3.0, max(0.0, p))
        hits = c["hits"]
        tri = 1.0 if c["triage_hits"] else 0.0
        return 2.0 * min(hits, 4) + 0.55 * p + 1.5 * tri

    def _find_cluster(self, x, y):
        best = None
        bd = 6.0
        for i, c in enumerate(self.clusters):
            if c["resolved"]:
                continue
            d = math.hypot(x - c["x"], y - c["y"])
            if d < bd:
                bd = d
                best = i
        return best

    def _add_detections(self, detections):
        for d in detections:
            x = float(d["x"])
            y = float(d["y"])
            tri = d.get("triage")
            k = self._find_cluster(x, y)
            if k is None:
                self.clusters.append({
                    "x": x, "y": y, "hits": 1,
                    "triage_hits": [tri] if tri in self.triage_levels else [],
                    "last_t": float(d.get("t", 0.0)),
                    "resolved": False,
                    "queued": False,
                })
            else:
                c = self.clusters[k]
                # Running mean reduces the already-small sensor position noise.
                n = c["hits"]
                c["x"] = (c["x"] * n + x) / (n + 1.0)
                c["y"] = (c["y"] * n + y) / (n + 1.0)
                c["hits"] = n + 1
                c["last_t"] = float(d.get("t", c["last_t"]))
                if tri in self.triage_levels:
                    c["triage_hits"].append(tri)

        # Queue strong candidates once.  A singleton is accepted only when
        # the prior is unusually high; otherwise require repeated detection.
        for c in self.clusters:
            if c["resolved"] or c["queued"]:
                continue
            p = self._cell_prior(c["x"], c["y"]) / self.prior_mean
            strong = c["hits"] >= 2
            very_high_prior = p >= 2.0 and c["hits"] >= 1
            if strong or very_high_prior or c["triage_hits"]:
                c["queued"] = True
                self.verify_queue.append(c)

    def _triage_choice(self, hits):
        if not hits:
            return None
        counts = {"critical": 0, "serious": 0, "minor": 0}
        for x in hits:
            if x in counts:
                counts[x] += 1
        return max(counts, key=counts.get)

    def _estimate_return_cost(self, obs):
        d = self._dist(obs["x"], obs["y"], self.base[0], self.base[1])
        duration = max(d / self.vmax,
                       abs(float(obs["altitude"]) - self.search_alt) / self.max_climb,
                       0.5)
        return duration * max(1.65, self.max_power_est * 1.15)

    def _safe_to_start(self, obs, target_x, target_y, target_alt):
        d = math.hypot(target_x - obs["x"], target_y - obs["y"])
        dur = max(d / self.vmax,
                  abs(target_alt - obs["altitude"]) / self.max_climb,
                  0.5)
        # Verification may need one hover and one declaration.
        projected = dur * 1.65 + 1.65 + 1.65
        return float(obs["energy_remaining"]) > self._estimate_return_cost(obs) + projected + 45.0

    def _choose_verification(self, obs):
        if not self.verify_queue:
            return None
        x, y = obs["x"], obs["y"]
        best = None
        best_score = -1e30
        for c in self.verify_queue:
            if c["resolved"]:
                continue
            d = math.hypot(c["x"] - x, c["y"] - y)
            # Prefer high-confidence candidates, but discount very expensive detours.
            s = self._candidate_score(c) / (1.0 + d / 120.0)
            if s > best_score:
                best_score = s
                best = c
        return best

    def _remove_from_queue(self, c):
        self.verify_queue = [q for q in self.verify_queue if q is not c]

    def _process_new_triage(self, c):
        if not c["triage_hits"]:
            return None
        return self._triage_choice(c["triage_hits"])

    def _mark_declared(self, c):
        c["resolved"] = True
        self.declared.append((c["x"], c["y"]))
        self._remove_from_queue(c)
        self.current_verify = None
        self.verify_stage = 0

    def act(self, obs):
        self._seen_action += 1

        # Online energy-rate estimate.  Keep a conservative upper envelope for
        # return decisions because wind can change.
        t = float(obs["t"])
        if self.last_t is not None:
            dt = t - self.last_t
            if dt > 1e-6:
                p = float(obs.get("energy_used_last_action", 0.0)) / dt
                if p > 0.0 and math.isfinite(p):
                    self.power_est = 0.8 * self.power_est + 0.2 * p
                    self.max_power_est = max(self.max_power_est * 0.995, p)
        self.last_t = t

        self._add_detections(obs.get("detections", []))

        # Never sacrifice the return.  The reserve is intentionally generous.
        ret = self._estimate_return_cost(obs)
        if (not self.returning and
                float(obs["energy_remaining"]) < ret + 55.0):
            self.returning = True

        if self.returning:
            if self._dist(obs["x"], obs["y"], self.base[0], self.base[1]) <= 12.0:
                return {"type": "land"}
            return {
                "type": "fly_to",
                "x": self.base[0], "y": self.base[1],
                "altitude": self.search_alt, "speed": self.vmax
            }

        # If a candidate is already at low altitude and has triage, declare it
        # immediately; declaration time matters for the exponential decay.
        if self.current_verify is not None:
            c = self.current_verify
            tri = self._process_new_triage(c)
            if tri is not None:
                self._mark_declared(c)
                return {"type": "declare", "x": c["x"], "y": c["y"], "triage": tri}

            if self.verify_stage == 1:
                # One short low-altitude hover gives another pair of detection
                # opportunities before we fall back to a conservative serious label.
                self.verify_stage = 2
                return {"type": "hover", "duration": 1.0}

            # A repeated high-altitude candidate with no low-altitude triage is
            # still much more likely real than a random false positive.
            if c["hits"] >= 2:
                self._mark_declared(c)
                return {"type": "declare", "x": c["x"], "y": c["y"], "triage": "serious"}

            c["resolved"] = True
            self._remove_from_queue(c)
            self.current_verify = None
            self.verify_stage = 0

        # During the sweep, verify a nearby strong candidate so the detour does
        # not destroy the route.  Otherwise keep covering new prior mass.
        candidate = self._choose_verification(obs)
        if candidate is not None:
            d = math.hypot(candidate["x"] - obs["x"], candidate["y"] - obs["y"])
            if d <= 110.0 and self._safe_to_start(obs, candidate["x"], candidate["y"], self.verify_alt):
                self.current_verify = candidate
                self.verify_stage = 1
                return {
                    "type": "fly_to",
                    "x": candidate["x"], "y": candidate["y"],
                    "altitude": self.verify_alt, "speed": self.verify_speed
                }

        # Once all coverage lanes are done, spend remaining safe budget on
        # strongest candidates, then return.
        if self.lane_i >= len(self.lane_order):
            self.search_done = True
            candidate = self._choose_verification(obs)
            if candidate is not None and self._safe_to_start(
                    obs, candidate["x"], candidate["y"], self.verify_alt):
                self.current_verify = candidate
                self.verify_stage = 1
                return {
                    "type": "fly_to",
                    "x": candidate["x"], "y": candidate["y"],
                    "altitude": self.verify_alt, "speed": self.verify_speed
                }
            self.returning = True
            return {
                "type": "fly_to",
                "x": self.base[0], "y": self.base[1],
                "altitude": self.search_alt, "speed": self.vmax
            }

        # Full-height search lane.  Alternating endpoints minimizes the
        # inter-lane transit while maintaining deterministic coverage.
        lane = self.lane_order[self.lane_i]
        self.lane_i += 1
        y = self.arena if self.go_up else 0.0
        self.go_up = not self.go_up
        return {
            "type": "fly_to",
            "x": self.lanes[lane], "y": y,
            "altitude": self.search_alt, "speed": self.search_speed
        }
