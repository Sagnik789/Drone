# dump_calibration.py
from sim import load_scenario
from sim.env import Episode

for seed in [9001, 9002, 9003]:

    scn = load_scenario(seed, is_calibration=True)
    ep = Episode(scn)

    obs = ep.observe()

    print("\n====================")
    print("SEED:", seed)
    print("====================")

    print("INFO:")
    print(ep.info())

    print("\nTRUTH:")
    print(obs["calibration_truth"])