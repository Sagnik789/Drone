import sys

if sys.version_info[:2] != (3, 10):
    raise RuntimeError("This simulator requires Python 3.10 (you are running %d.%d). See README.md, section 1."
                       % sys.version_info[:2])

from .scenarios import load_scenario, scenario_seeds
from .runner import run_episode

__all__ = ["load_scenario", "scenario_seeds", "run_episode"]
