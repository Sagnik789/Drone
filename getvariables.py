# extract_physics.py
#
# Run from repository root with Python 3.10:
#     python extract_physics.py
#
# Outputs:
#     physics_results.csv
#     calibration_dump.txt
#
# Uses ONLY the simulator's public API + calibration_truth.
# It does not modify the simulator.

from sim import load_scenario
from sim.env import Episode
import csv
import math
import statistics
import random


SEEDS = [9001, 9002, 9003]

ALTITUDES = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80]
SPEEDS = [2, 5, 8, 10, 12, 15]
DISTANCES = [25, 50, 100, 150]
HOVER_TIMES = [1, 2, 5, 10, 20]

# Number of repeated experiments.
REPEATS = 5

ROWS = []


def run_episode(seed):
    return Episode(load_scenario(seed, is_calibration=True))


def get_truth(obs):
    return obs.get("calibration_truth", {})


def survivor_xy(s):
    if isinstance(s, dict):
        return s.get("x"), s.get("y")
    return s[0], s[1]


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def add(row):
    ROWS.append(row)


# ============================================================
# 1. CALIBRATION TRUTH / CONSTANTS
# ============================================================

with open("calibration_dump.txt", "w", encoding="utf-8") as f:

    for seed in SEEDS:
        ep = run_episode(seed)
        obs = ep.observe()

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"SEED {seed}\n")
        f.write("=" * 80 + "\n")

        f.write("\nINFO:\n")
        f.write(repr(ep.info()))

        f.write("\n\nCALIBRATION TRUTH:\n")
        f.write(repr(get_truth(obs)))
        f.write("\n")

        truth = get_truth(obs)

        f.write("\nSURVIVORS:\n")
        for s in truth.get("survivors", []):
            f.write(repr(s) + "\n")

        f.write("\nWIND:\n")
        f.write(repr(truth.get("wind")) + "\n")


# ============================================================
# 2. ENERGY VS DISTANCE / SPEED / ALTITUDE
# ============================================================

print("Running energy experiments...")

for seed in SEEDS:

    for altitude in ALTITUDES:
        for speed in SPEEDS:
            for distance_m in DISTANCES:

                ep = run_episode(seed)

                ep.step({
                    "action": "fly_to",
                    "x": distance_m,
                    "y": 0,
                    "altitude": altitude,
                    "speed": speed,
                })

                obs = ep.observe()

                add({
                    "experiment": "energy",
                    "seed": seed,
                    "altitude": altitude,
                    "speed": speed,
                    "distance": distance_m,
                    "duration": obs["t"],
                    "energy": obs["energy_used_last_action"],
                    "n_detections": len(obs.get("detections", [])),
                })


# ============================================================
# 3. ENERGY VS HEADING / WIND
# ============================================================

print("Running wind/heading experiments...")

HEADINGS = {
    "E": (100, 0),
    "W": (-100, 0),
    "N": (0, 100),
    "S": (0, -100),
    "NE": (100, 100),
    "NW": (-100, 100),
    "SE": (100, -100),
    "SW": (-100, -100),
}

for seed in SEEDS:

    for heading, (x, y) in HEADINGS.items():

        ep = run_episode(seed)

        ep.step({
            "action": "fly_to",
            "x": x,
            "y": y,
            "altitude": 40,
            "speed": 10,
        })

        obs = ep.observe()
        truth = get_truth(obs)

        add({
            "experiment": "heading",
            "seed": seed,
            "heading": heading,
            "altitude": 40,
            "speed": 10,
            "distance": math.hypot(x, y),
            "duration": obs["t"],
            "energy": obs["energy_used_last_action"],
            "wind_x": (
                truth.get("wind", [None, None])[0]
                if truth.get("wind") else None
            ),
            "wind_y": (
                truth.get("wind", [None, None])[1]
                if truth.get("wind") else None
            ),
        })


# ============================================================
# 4. DETECTION VS ALTITUDE / HOVER TIME
# ============================================================

print("Running detection experiments...")

# We deliberately repeat the same experiment to estimate probability.

for seed in SEEDS:

    # Find a location containing at least one survivor.
    base_ep = run_episode(seed)
    base_truth = get_truth(base_ep.observe())
    survivors = base_truth.get("survivors", [])

    if not survivors:
        continue

    sx, sy = survivor_xy(survivors[0])

    for altitude in ALTITUDES:
        for duration in HOVER_TIMES:

            for repeat in range(REPEATS):

                ep = run_episode(seed)

                ep.step({
                    "action": "fly_to",
                    "x": sx,
                    "y": sy,
                    "altitude": altitude,
                    "speed": 10,
                })

                ep.step({
                    "action": "hover",
                    "duration": duration,
                })

                obs = ep.observe()

                detections = obs.get("detections", [])

                # Count detections close to the known survivor.
                found = any(
                    distance(
                        (d["x"], d["y"]),
                        (sx, sy)
                    ) <= 5
                    for d in detections
                )

                add({
                    "experiment": "detection",
                    "seed": seed,
                    "altitude": altitude,
                    "duration": duration,
                    "speed": 10,
                    "survivor_x": sx,
                    "survivor_y": sy,
                    "detected": int(found),
                    "n_detections": len(detections),
                })


# ============================================================
# 5. POSITION ERROR
# ============================================================

print("Running position-noise experiments...")

for seed in SEEDS:

    ep0 = run_episode(seed)
    truth = get_truth(ep0.observe())
    survivors = truth.get("survivors", [])

    if not survivors:
        continue

    for altitude in ALTITUDES:

        # Test each survivor individually.
        for survivor in survivors:

            sx, sy = survivor_xy(survivor)

            ep = run_episode(seed)

            ep.step({
                "action": "fly_to",
                "x": sx,
                "y": sy,
                "altitude": altitude,
                "speed": 10,
            })

            ep.step({
                "action": "hover",
                "duration": 20,
            })

            obs = ep.observe()

            for d in obs.get("detections", []):

                err = distance(
                    (d["x"], d["y"]),
                    (sx, sy)
                )

                add({
                    "experiment": "position_error",
                    "seed": seed,
                    "altitude": altitude,
                    "true_x": sx,
                    "true_y": sy,
                    "reported_x": d["x"],
                    "reported_y": d["y"],
                    "error": err,
                })


# ============================================================
# 6. FALSE POSITIVE / CLUTTER MEASUREMENT
# ============================================================

print("Running false-positive experiments...")

for seed in SEEDS:

    ep0 = run_episode(seed)
    truth = get_truth(ep0.observe())
    survivors = truth.get("survivors", [])

    truth_xy = [
        survivor_xy(s)
        for s in survivors
    ]

    # Test center and several locations.
    locations = [
        (0, 0),
        (100, 0),
        (-100, 0),
        (0, 100),
        (0, -100),
        (150, 150),
        (-150, -150),
    ]

    for x, y in locations:

        for altitude in ALTITUDES:

            ep = run_episode(seed)

            ep.step({
                "action": "fly_to",
                "x": x,
                "y": y,
                "altitude": altitude,
                "speed": 10,
            })

            ep.step({
                "action": "hover",
                "duration": 20,
            })

            obs = ep.observe()

            false_count = 0

            for d in obs.get("detections", []):

                dx = d["x"]
                dy = d["y"]

                if not truth_xy:
                    false_count += 1
                    continue

                if min(
                    distance((dx, dy), s)
                    for s in truth_xy
                ) > 5:
                    false_count += 1

            add({
                "experiment": "false_positive",
                "seed": seed,
                "x": x,
                "y": y,
                "altitude": altitude,
                "duration": 20,
                "n_detections": len(obs.get("detections", [])),
                "false_detections": false_count,
            })


# ============================================================
# 7. TRIAGE AVAILABILITY / ACCURACY
# ============================================================

print("Running triage experiments...")

for seed in SEEDS:

    ep0 = run_episode(seed)
    truth = get_truth(ep0.observe())
    survivors = truth.get("survivors", [])

    for altitude in ALTITUDES:

        for survivor in survivors:

            sx, sy = survivor_xy(survivor)

            # Obtain repeated observations near survivor.
            ep = run_episode(seed)

            ep.step({
                "action": "fly_to",
                "x": sx,
                "y": sy,
                "altitude": altitude,
                "speed": 5,
            })

            ep.step({
                "action": "hover",
                "duration": 10,
            })

            obs = ep.observe()

            for d in obs.get("detections", []):

                add({
                    "experiment": "triage",
                    "seed": seed,
                    "altitude": altitude,
                    "true_x": sx,
                    "true_y": sy,
                    "true_triage": (
                        survivor.get("triage")
                        if isinstance(survivor, dict)
                        else None
                    ),
                    "reported_triage": d.get("triage"),
                })


# ============================================================
# 8. WRITE EVERYTHING TO CSV
# ============================================================

print("Writing physics_results.csv...")

if ROWS:

    # Collect every key appearing in any experiment.
    columns = []

    for row in ROWS:
        for key in row:
            if key not in columns:
                columns.append(key)

    with open(
        "physics_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(ROWS)


# ============================================================
# 9. PRINT QUICK SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHYSICS EXPERIMENT COMPLETE")
print("=" * 70)

print(f"Total measurements: {len(ROWS)}")
print("Files:")
print("  calibration_dump.txt")
print("  physics_results.csv")

print("\nCalibration wind:")
for seed in SEEDS:
    ep = run_episode(seed)
    truth = get_truth(ep.observe())
    print(
        f"  {seed}: wind={truth.get('wind')}"
    )

# Detection summary
detection_rows = [
    r for r in ROWS
    if r["experiment"] == "detection"
]

if detection_rows:
    print("\nDetection rates:")

    for altitude in ALTITUDES:
        for duration in HOVER_TIMES:

            subset = [
                r for r in detection_rows
                if r.get("altitude") == altitude
                and r.get("duration") == duration
            ]

            if subset:
                rate = sum(
                    r["detected"]
                    for r in subset
                ) / len(subset)

                print(
                    f"  altitude={altitude:2} "
                    f"time={duration:2} "
                    f"P(detect)={rate:.4f}"
                )

# Position error summary
error_rows = [
    r for r in ROWS
    if r["experiment"] == "position_error"
]

if error_rows:
    print("\nPosition error:")

    for altitude in ALTITUDES:

        vals = [
            r["error"]
            for r in error_rows
            if r.get("altitude") == altitude
        ]

        if vals:
            p95_index = min(
            len(vals) - 1,
            int(0.95 * len(vals))
            )

            p95 = sorted(vals)[p95_index]

            print(
                f"  altitude={altitude:2} "
                f"N={len(vals):4} "
                f"mean={statistics.mean(vals):.3f} "
                f"median={statistics.median(vals):.3f} "
                f"p95={p95:.3f}"
            )

# Energy summary
energy_rows = [
    r for r in ROWS
    if r["experiment"] == "energy"
]

if energy_rows:
    print("\nSample energy measurements:")

    for r in energy_rows[:20]:
        print(
            f"  h={r['altitude']:2} "
            f"v={r['speed']:2} "
            f"d={r['distance']:3} "
            f"E={r['energy']:.6f} "
            f"t={r['duration']:.3f}"
        )

print("\nUse physics_results.csv for fitting formulas.")
print("Give calibration_dump.txt + physics_results.csv to Codex.")