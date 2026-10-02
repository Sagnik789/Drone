import math

ALT = 40.0
SPEED = 10.0
DEDUP_RADIUS = 15.0
CLUSTER_RADIUS = 6.5
CONFIRM_RADIUS = 9.0
SINGLETON_WAIT = 1
SINGLETON_PRIOR_MULT = 2.0
SEGMENT = 50.0


class Policy:
    def __init__(self, info):
        self.arena = float(info["arena_size"])
        self.budget = float(info["energy_budget"])
        self.prior = info["prior_map"]
        self.g = int(info["grid_size"])
        self.cell = float(info["cell_size"])
        self.fov = float(
            info.get("footprint_tan", math.tan(math.radians(40.0)))
        )

        # Exact proven lawnmower geometry.
        r = ALT * self.fov
        spacing = 1.6 * r
        self.waypoints = []
        x, up = spacing / 2.0, True
        n = int(math.ceil(self.arena / SEGMENT))

        while x < self.arena:
            ys = [0.0, self.arena] if up else [self.arena, 0.0]
            for i in range(n + 1):
                self.waypoints.append(
                    (x, ys[0] + (ys[1] - ys[0]) * i / n)
                )
            x += spacing
            up = not up

        self.wp = 0
        self.declared = []
        self.queue = []
        self.pending = []
        self.returning = False
        self.power_est = 2.0
        self.last = None

        total = sum(sum(float(v) for v in row) for row in self.prior)
        self.mean_prior = total / max(1, self.g * self.g)

    def _prior(self, x, y):
        ix = int(x / self.cell)
        iy = int(y / self.cell)
        if 0 <= ix < self.g and 0 <= iy < self.g:
            return float(self.prior[ix][iy])
        return 0.0

    def _near_declared(self, x, y):
        return any(
            math.hypot(x - px, y - py) <= DEDUP_RADIUS
            for px, py in self.declared
        )

    def _cluster(self, detections):
        pts = [(float(d["x"]), float(d["y"])) for d in detections]
        unused = set(range(len(pts)))
        clusters = []

        while unused:
            seed = min(unused)
            unused.remove(seed)
            members = [seed]

            changed = True
            while changed:
                changed = False
                cx = sum(pts[i][0] for i in members) / len(members)
                cy = sum(pts[i][1] for i in members) / len(members)

                for i in list(unused):
                    if (
                        math.hypot(
                            pts[i][0] - cx,
                            pts[i][1] - cy,
                        )
                        <= CLUSTER_RADIUS
                    ):
                        members.append(i)
                        unused.remove(i)
                        changed = True

            # Robust against the calibration's high-altitude outlier mode.
            sx = sorted(pts[i][0] for i in members)
            sy = sorted(pts[i][1] for i in members)
            mid = len(members) // 2

            if len(members) % 2:
                cx, cy = sx[mid], sy[mid]
            else:
                cx = 0.5 * (sx[mid - 1] + sx[mid])
                cy = 0.5 * (sy[mid - 1] + sy[mid])

            clusters.append((cx, cy, len(members)))

        return clusters

    def _ingest(self, obs):
        # Current action detections.
        current = []
        for d in obs.get("detections", []):
            try:
                x = float(d["x"])
                y = float(d["y"])
            except (KeyError, TypeError, ValueError):
                continue

            if not self._near_declared(x, y):
                current.append((x, y))

        # Cluster detections from this action exactly as v2 did.
        detections = [{"x": x, "y": y} for x, y in current]
        clusters = self._cluster(detections) if detections else []

        strong = [c for c in clusters if c[2] >= 2]

        # Strong same-action clusters are immediately accepted.
        chosen = list(strong)

        # Merge current singletons with pending singletons from earlier
        # actions. A nearby repeat confirms the candidate.
        new_pending = []

        for c in clusters:
            if c[2] >= 2:
                continue

            x, y, _ = c

            match = None
            best = CONFIRM_RADIUS + 1.0
            for p in self.pending:
                dist = math.hypot(x - p[0], y - p[1])
                if dist <= CONFIRM_RADIUS and dist < best:
                    match = p
                    best = dist

            if match is not None:
                # Confirmed by a separate observation.
                cx = 0.5 * (x + match[0])
                cy = 0.5 * (y + match[1])
                chosen.append((cx, cy, 2))
            else:
                # Keep singleton alive for a short confirmation window.
                new_pending.append((x, y, match[2] + 1 if match else 1))

        # Age old pending candidates. High-prior singletons are allowed
        # through after waiting; weak unconfirmed singletons are discarded.
        for x, y, age in self.pending:
            if any(
                math.hypot(x - nx, y - ny) <= CONFIRM_RADIUS
                for nx, ny, _ in new_pending
            ):
                continue

            if age + 1 >= SINGLETON_WAIT:
                if self._prior(x, y) >= SINGLETON_PRIOR_MULT * self.mean_prior:
                    chosen.append((x, y, 1))
            else:
                new_pending.append((x, y, age + 1))

        self.pending = new_pending

        # Avoid duplicate declarations/queued candidates.
        for x, y, count in chosen:
            if self._near_declared(x, y):
                continue

            duplicate = False
            for qx, qy, _ in self.queue:
                if math.hypot(x - qx, y - qy) <= DEDUP_RADIUS:
                    duplicate = True
                    break

            if not duplicate:
                self.queue.append((x, y, count))

    def act(self, obs):
        x = float(obs["x"])
        y = float(obs["y"])

        if (
            self.last is not None
            and self.last[0] == "fly_to"
            and float(obs["t"]) > self.last[1]
        ):
            dt = float(obs["t"]) - self.last[1]
            if dt > 0:
                p = float(obs.get("energy_used_last_action", 0.0)) / dt
                if p > 0.0 and math.isfinite(p):
                    self.power_est = 0.7 * self.power_est + 0.3 * p

        self._ingest(obs)

        if self.queue:
            self.queue.sort(
                key=lambda c: (
                    -c[2],
                    math.hypot(c[0] - x, c[1] - y),
                )
            )

            px, py, _ = self.queue.pop(0)
            self.declared.append((px, py))
            self.last = ("declare", float(obs["t"]))

            return {
                "type": "declare",
                "x": px,
                "y": py,
                "triage": "serious",
            }

        home = math.hypot(x, y)

        if not self.returning:
            ret_cost = (
                home / SPEED * self.power_est * 1.5
                + 0.05 * self.budget
            )

            if (
                float(obs["energy_remaining"]) < ret_cost
                or self.wp >= len(self.waypoints)
            ):
                self.returning = True

        if self.returning:
            if home <= 10.0:
                return {"type": "land"}

            self.last = ("fly_to", float(obs["t"]))

            return {
                "type": "fly_to",
                "x": 0.0,
                "y": 0.0,
                "altitude": ALT,
                "speed": SPEED,
            }

        tx, ty = self.waypoints[self.wp]
        self.wp += 1
        self.last = ("fly_to", float(obs["t"]))

        return {
            "type": "fly_to",
            "x": tx,
            "y": ty,
            "altitude": ALT,
            "speed": SPEED,
        }
