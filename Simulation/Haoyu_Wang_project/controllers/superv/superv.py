"""Webots Supervisor batch runner for UR5e pick-and-place experiments.

- Respawns red/blue cubes each episode.
- Sends start commands to robot controller.
- Logs results to results.csv.
"""

from controller import Supervisor
import csv
import json
import math
import random
import time

# ----------------------------
# Supervisor batch runner
# ----------------------------
sup = Supervisor()
timestep = int(sup.getBasicTimeStep())

tx = sup.getDevice("tx")
rx = sup.getDevice("rx")
rx.enable(timestep)

# World nodes (must exist as DEFs in the .wbt)
red = sup.getFromDef("RED_CUBE")
blue = sup.getFromDef("BLUE_CUBE")
light = sup.getFromDef("MAIN_LIGHT")  # TexturedBackground in your world

if red is None or blue is None or light is None:
    raise RuntimeError("Missing DEFs: need RED_CUBE, BLUE_CUBE, MAIN_LIGHT in the world.")

# Fields
red_t = red.getField("translation")
red_r = red.getField("rotation")
blue_t = blue.getField("translation")
blue_r = blue.getField("rotation")

# NOTE: your world uses TexturedBackground { luminosity ... }
lum_field = light.getField("luminosity")

# Spawn region (from your previous bounds)
xmin, xmax = -0.28, 0.08
ymin, ymax =  2.62, 2.88
z0 = 0.7149541003

cube_size = 0.03
margin = cube_size / 2.0

def sample_xy():
    x = random.uniform(xmin + margin, xmax - margin)
    y = random.uniform(ymin + margin, ymax - margin)
    return x, y

def respawn_two_cubes(min_dist=0.06, max_tries=200):
    rx0, ry0 = sample_xy()
    for _ in range(max_tries):
        bx0, by0 = sample_xy()
        if (bx0 - rx0)**2 + (by0 - ry0)**2 >= min_dist**2:
            break
    else:
        bx0, by0 = sample_xy()

    r_yaw = random.uniform(0.0, 2.0 * math.pi)
    b_yaw = random.uniform(0.0, 2.0 * math.pi)

    red_t.setSFVec3f([rx0, ry0, z0])
    red_r.setSFRotation([0, 0, 1, r_yaw])
    red.resetPhysics()

    blue_t.setSFVec3f([bx0, by0, z0])
    blue_r.setSFRotation([0, 0, 1, b_yaw])
    blue.resetPhysics()

def send_start(trial_id: int, illum: float, seed: int, target_color: str = None):
    tx.send(json.dumps({
        "cmd": "start",
        "trial_id": int(trial_id),
        "illum": float(illum),
        "seed": int(seed),
        "target_color": target_color,
    }).encode("utf-8"))

# ----------------------------
# Experiment config
# ----------------------------
illum_levels = [0.000002]
episodes_per_level = 10
timeout_s = 900.0  # per trial wall-clock timeout (avoid deadlock)

# CSV
f = open("results.csv", "w", newline="")
writer = csv.DictWriter(f, fieldnames=[
    "trial_id", "illum", "episode", "target_color",
    "success", "success_gt", "reason",
    "sim_time", "wall_time",
    "ret_calls", "ret_wall",
    "picked_color",
])
writer.writeheader()


trial_id = 0

# Small helper to drain stale packets
def drain_rx(max_packets=50):
    k = 0
    while rx.getQueueLength() > 0 and k < max_packets:
        rx.nextPacket()
        k += 1

# Main loop
for illum in illum_levels:
    # set illumination once per level
    lum_field.setSFFloat(float(illum))

    for ep in range(episodes_per_level):
        seed = 2003 + ep
        random.seed(seed)

        # respawn BOTH cubes once per episode (then we will pick red, then blue)
        respawn_two_cubes()

        for target in ("red", "blue"):
            trial_id += 1

            # clear old messages, then start
            drain_rx()
            send_start(trial_id, illum, seed, target_color=target)

            # wait for done
            t0 = time.perf_counter()
            got = False

            while sup.step(timestep) != -1:
                if time.perf_counter() - t0 > timeout_s:
                    writer.writerow({
                        "trial_id": trial_id,
                        "illum": illum,
                        "episode": ep,
                        "target_color": target,
                        "success": 0,
                        "success_gt": 0,
                        "reason": "timeout",
                        "sim_time": None,
                        "wall_time": None,
                        "ret_calls": None,
                        "ret_wall": None,
                        "picked_color": None,
                    })
                    f.flush()
                    got = True
                    break

                if rx.getQueueLength() == 0:
                    continue

                msg = json.loads(rx.getString())
                rx.nextPacket()

                picked_color = msg.get("picked_color", None)

                if int(msg.get("trial_id", -999)) != int(trial_id):
                    # stale packet from previous trial; ignore
                    continue

                # ----------------------------
                # Ground-truth success (Supervisor): did the TARGET cube move into the correct bin side?
                #   - red bin is on the LEFT (x ~ -0.75)
                #   - blue bin is on the RIGHT (x ~ +0.65)
                # You can tighten thresholds later.
                # ----------------------------
                if picked_color == "red":
                    cube_pos = red_t.getSFVec3f()
                    success_gt = 1 if cube_pos[0] < -0.45 else 0
                elif picked_color == "blue":
                    cube_pos = blue_t.getSFVec3f()
                    success_gt = 1 if cube_pos[0] > 0.30 else 0
                else:
                    success_gt = 0

                success_ctrl = int(msg.get("success", 0))
                success_final = 1 if (success_ctrl == 1 and success_gt == 1) else 0

                writer.writerow({
                    "trial_id": trial_id,
                    "illum": illum,
                    "episode": ep,
                    "target_color": target,
                    "success": success_final,
                    "success_gt": success_gt,
                    "reason": msg.get("reason"),
                    "sim_time": msg.get("sim_time"),
                    "wall_time": msg.get("wall_time"),
                    "ret_calls": msg.get("ret_calls"),
                    "ret_wall": msg.get("ret_wall"),
                    "picked_color": msg.get("picked_color"),
                })
                f.flush()
                got = True
                break

            if not got:
                # simulation ended unexpectedly
                break

f.close()
sup.simulationQuit(0)
