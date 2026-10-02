from sim import load_scenario
from sim.env import Episode

ep = Episode(load_scenario(9001, is_calibration=True))

print("INITIAL")
print(ep.observe())

ep.step({
    "type": "fly_to",
    "x": 100,
    "y": 0,
    "altitude": 40,
    "speed": 10
})

print("\nAFTER FLIGHT")
print(ep.observe())

ep.step({
    "type": "hover",
    "duration": 10
})

print("\nAFTER HOVER")
print(ep.observe())