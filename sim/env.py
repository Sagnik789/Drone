import math
import numbers

import numpy as np

from . import _core
from .scoring import CONFIG, score_episode

ARENA_SIZE = _core.ARENA
GRID_SIZE = _core.GRID
CELL_SIZE = _core.CELL
BASE_XY = _core.BASE
ALTITUDE_RANGE = _core.ALT_RANGE
SPEED_RANGE = _core.SPEED_RANGE
MAX_CLIMB_RATE = _core.MAX_CLIMB_RATE
START_ALTITUDE = _core.START_ALTITUDE
FOOTPRINT_TAN = _core.TAN_FOV
HOVER_RANGE = (0.5, 30.0)
DECLARE_DURATION = 1.0
LAND_RADIUS = 15.0
MIN_MOVE_DURATION = 0.5
TRIAGE_LEVELS = _core.TRIAGE_LEVELS


def _num(v):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _clamp(v, lo, hi):
    return min(max(v, lo), hi)


class Episode:
    def __init__(self, scenario, config=None, record_trace=False):
        self.scn = scenario
        self.cfg = config or CONFIG
        self.limits = self.cfg["limits"]
        self.state = _core.initial_state(scenario)
        self.n_actions = 0
        self.n_invalid = 0
        self.declarations = []
        self.done = False
        self.landed = False
        self.end_reason = None
        self._last_energy = 0.0
        self._last_status = "ok"
        self._last_detections = []
        self.record_trace = record_trace
        s = self.state
        self.trace = [np.array([[s.t, s.x, s.y, s.h]])] if record_trace else None
        self.all_detections = [] if record_trace else None

    def info(self):
        c = self.cfg
        return {
            "arena_size": ARENA_SIZE,
            "grid_size": GRID_SIZE,
            "cell_size": CELL_SIZE,
            "base_xy": list(BASE_XY),
            "prior_map": self.scn.prior.tolist(),
            "energy_budget": self.scn.energy_budget,
            "altitude_range": list(ALTITUDE_RANGE),
            "speed_range": list(SPEED_RANGE),
            "hover_range": list(HOVER_RANGE),
            "max_climb_rate": MAX_CLIMB_RATE,
            "start_altitude": START_ALTITUDE,
            "footprint_tan": FOOTPRINT_TAN,
            "land_radius": LAND_RADIUS,
            "declare_duration": DECLARE_DURATION,
            "triage_weights": dict(c["triage_weights"]),
            "triage_decay_tau": dict(c["triage_decay_tau"]),
            "wrong_triage_factor": c["wrong_triage_factor"],
            "match_radius": c["match_radius"],
            "lambda_fp": c["lambda_fp"],
            "return_penalty": c["return_penalty"],
            "max_actions": self.limits["max_actions"],
            "act_time_limit_s": self.limits["act_time_limit_s"],
            "episode_decision_time_limit_s": self.limits["episode_decision_time_limit_s"],
            "is_calibration": self.scn.is_calibration,
        }

    def observe(self):
        s = self.state
        obs = {
            "t": round(s.t, 4),
            "x": round(s.x, 4),
            "y": round(s.y, 4),
            "altitude": round(s.h, 4),
            "energy_remaining": round(s.energy, 4),
            "energy_used_last_action": round(self._last_energy, 4),
            "last_action_status": self._last_status,
            "detections": [{"x": round(x, 3), "y": round(y, 3), "triage": tri, "t": round(t, 3)}
                           for (t, x, y, tri) in self._last_detections],
            "n_declared": len(self.declarations),
            "n_actions": self.n_actions,
        }
        if self.scn.is_calibration:
            obs["calibration_truth"] = _core.calibration_truth(self.scn, s.t)
        return obs

    def _move(self, x1, y1, h1, duration):
        out = _core.simulate(self.scn, self.state, x1, y1, h1, duration, self.record_trace)
        self._last_energy = out["energy_used"]
        self._last_detections = out["detections"]
        if self.record_trace:
            self.trace.append(out["trace"])
            self.all_detections.extend(out["detections"])
        if out["depleted"]:
            self.end("energy_depleted")

    def _invalid(self):
        self._last_status = "invalid"
        self._last_energy = 0.0
        self._last_detections = []
        self.n_invalid += 1
        if self.n_invalid >= self.limits["max_invalid_actions"]:
            self.end("too_many_invalid")

    def step(self, action):
        if self.done:
            return
        self.n_actions += 1
        self._apply(action)
        if not self.done and self.n_actions >= self.limits["max_actions"]:
            self.end("max_actions")

    def _apply(self, a):
        if not isinstance(a, dict):
            return self._invalid()
        kind = a.get("type")
        s = self.state
        clamped = False

        if kind == "fly_to":
            x, y, h, v = (_num(a.get(k)) for k in ("x", "y", "altitude", "speed"))
            if None in (x, y, h, v):
                return self._invalid()
            cx, cy = _clamp(x, 0.0, ARENA_SIZE), _clamp(y, 0.0, ARENA_SIZE)
            ch = _clamp(h, *ALTITUDE_RANGE)
            cv = _clamp(v, *SPEED_RANGE)
            clamped = (cx, cy, ch, cv) != (x, y, h, v)
            dist = math.hypot(cx - s.x, cy - s.y)
            duration = max(dist / cv, abs(ch - s.h) / MAX_CLIMB_RATE, MIN_MOVE_DURATION)
            self._last_status = "clamped" if clamped else "ok"
            self._move(cx, cy, ch, duration)

        elif kind == "hover":
            d = _num(a.get("duration"))
            if d is None:
                return self._invalid()
            cd = _clamp(d, *HOVER_RANGE)
            self._last_status = "clamped" if cd != d else "ok"
            self._move(s.x, s.y, s.h, cd)

        elif kind == "declare":
            x, y = _num(a.get("x")), _num(a.get("y"))
            tri = a.get("triage")
            if x is None or y is None or tri not in TRIAGE_LEVELS:
                return self._invalid()
            cx, cy = _clamp(x, 0.0, ARENA_SIZE), _clamp(y, 0.0, ARENA_SIZE)
            self._last_status = "clamped" if (cx, cy) != (x, y) else "ok"
            self.declarations.append((s.t, cx, cy, tri))
            self._move(s.x, s.y, s.h, DECLARE_DURATION)

        elif kind == "land":
            if math.hypot(s.x - BASE_XY[0], s.y - BASE_XY[1]) <= LAND_RADIUS:
                self._last_status = "ok"
                self._last_energy = 0.0
                self._last_detections = []
                self.landed = True
                self.end("landed")
            else:
                return self._invalid()
        else:
            return self._invalid()

    def end(self, reason):
        if not self.done:
            self.done = True
            self.end_reason = reason

    def result(self):
        xy, tri = _core.truth(self.scn)
        res = score_episode(self.declarations, xy, tri, self.landed, self.cfg)
        res.update({
            "scenario_id": self.scn.scenario_id,
            "end_reason": self.end_reason,
            "n_actions": self.n_actions,
            "n_invalid": self.n_invalid,
            "t_end": self.state.t,
            "energy_left": self.state.energy,
        })
        return res

    def trace_array(self):
        return np.concatenate(self.trace, axis=0) if self.trace else None
