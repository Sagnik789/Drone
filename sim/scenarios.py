import json
import pathlib

from . import _core

_SEED_FILE = pathlib.Path(__file__).resolve().parents[1] / "scenarios" / "seeds.json"


def scenario_seeds(which="public"):
    seeds = json.loads(_SEED_FILE.read_text())
    out = []
    if which in ("public", "all"):
        out += [(s, False) for s in seeds["public"]]
    if which in ("calibration", "all"):
        out += [(s, True) for s in seeds["calibration"]]
    if not out:
        raise ValueError("unknown scenario set %r" % which)
    return out


def load_scenario(seed, is_calibration=None):
    if is_calibration is None:
        is_calibration = seed in json.loads(_SEED_FILE.read_text())["calibration"]
    scn = _core.generate(seed, "local", is_calibration=is_calibration)
    scn.scenario_id = ("calib-%d" if is_calibration else "public-%d") % seed
    return scn
