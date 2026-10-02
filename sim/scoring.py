# Episode scoring, identical to the evaluation.
import json
import math
import pathlib


def _config_path():
    here = pathlib.Path(__file__).resolve()
    local = here.with_name("scoring.json")
    if local.exists():
        return local
    return here.parents[2] / "config" / "scoring.json"


def load_config(path=None):
    with open(path or _config_path()) as f:
        return json.load(f)


CONFIG = load_config()


def score_episode(declarations, survivors, triage, landed, config=None):
    cfg = config or CONFIG
    weights = cfg["triage_weights"]
    taus = cfg["triage_decay_tau"]
    radius = cfg["match_radius"]

    claimed = [False] * len(survivors)
    value = 0.0
    n_correct = n_false = n_right_triage = 0
    per_decl = []
    for (t, x, y, tri) in declarations:
        best, best_d = None, None
        for j, (sx, sy) in enumerate(survivors):
            if claimed[j]:
                continue
            d = math.hypot(x - sx, y - sy)
            if d <= radius and (best_d is None or d < best_d):
                best, best_d = j, d
        if best is None:
            n_false += 1
            per_decl.append({"t": t, "x": x, "y": y, "triage": tri, "correct": False, "value": 0.0})
            continue
        claimed[best] = True
        n_correct += 1
        true_tri = triage[best]
        factor = 1.0 if tri == true_tri else cfg["wrong_triage_factor"]
        n_right_triage += tri == true_tri
        v = weights[true_tri] * math.exp(-t / taus[true_tri]) * factor
        value += v
        per_decl.append({"t": t, "x": x, "y": y, "triage": tri, "correct": True, "value": v,
                         "survivor": best, "true_triage": true_tri})

    n_surv = len(survivors)
    total_w = sum(weights[tr] for tr in triage)
    value_norm = value / total_w if total_w > 0 else 0.0
    fp_term = cfg["lambda_fp"] * n_false / max(n_surv, 1)
    ret_pen = 0.0 if landed else cfg["return_penalty"]
    score = max(cfg["episode_score_floor"], value_norm - fp_term - ret_pen)
    return {
        "score": score,
        "value_norm": value_norm,
        "fp_term": fp_term,
        "return_penalty": ret_pen,
        "n_declared": len(declarations),
        "n_correct": n_correct,
        "n_right_triage": n_right_triage,
        "n_false": n_false,
        "n_survivors": n_surv,
        "landed": bool(landed),
        "declarations": per_decl,
    }
