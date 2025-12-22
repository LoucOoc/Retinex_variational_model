"""ur5e_controller controller (robot-side).

- Waits for Supervisor command {cmd:'start', trial_id, illum, seed}
- Runs ONE pick-and-place episode
- Sends back {event:'done', trial_id, success, ...metrics...}

NOTE:
Webots' Python import path can sometimes pick up an unexpected `control` module.
To make this robust, we explicitly load `control.py` from the SAME directory as
this controller file.
"""

from controller import Robot
import numpy as np
import json
import os
import importlib.util

# -----------------------------------------------------------------------------
# Robust local import of control.py (same folder as this file)
# -----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(__file__)
_control_path = os.path.join(_THIS_DIR, "control.py")
_spec = importlib.util.spec_from_file_location("control_local", _control_path)
_control = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader, f"Failed to load control.py from {_control_path}"
_spec.loader.exec_module(_control)

RobotPlacerWithVision = _control.RobotPlacerWithVision
initial_pose = _control.initial_pose

# Optional: sanity check (helps catch wrong file versions immediately)
if not hasattr(RobotPlacerWithVision, "getRobotCommand"):
    raise AttributeError(
        "RobotPlacerWithVision in local control.py is missing getRobotCommand(). "
        "Make sure controllers/ur5e_controller/control.py is the updated version."
    )

# ===== Optional: live preview with OpenCV =====
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    print("[camera] OpenCV not available; will skip live preview.")

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Comms (must exist as devices named tx/rx on the UR5e)
tx = robot.getDevice("tx")
rx = robot.getDevice("rx")
rx.enable(timestep)

# -----------------------------------------------------------------------------
# UR5e arm joints (6-DoF) + position sensors
# -----------------------------------------------------------------------------
arm_joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
motors = [robot.getDevice(n) for n in arm_joint_names]
sensors = []
for m in motors:
    ps = m.getPositionSensor()
    ps.enable(timestep)
    sensors.append(ps)

# -----------------------------------------------------------------------------
# Robotiq 2F-85 gripper
# -----------------------------------------------------------------------------
GRIP_MIN = 0.0
GRIP_MAX = 0.80

def _safe_get_device(name):
    try:
        return robot.getDevice(name)
    except Exception:
        return None

gripper_left  = _safe_get_device("robotiq_2f85::left finger joint")
gripper_right = _safe_get_device("robotiq_2f85::right finger joint")

if gripper_left and gripper_right:
    for g in (gripper_left, gripper_right):
        try:
            g.setVelocity(1.5)
        except Exception:
            pass
        try:
            g.setForce(50.0)
        except Exception:
            pass
    print("[gripper] Robotiq 2F-85 finger motors found.")
else:
    print("[gripper] Could not find Robotiq 2F-85 motors. "
          "Expected: 'robotiq_2f85::left finger joint' and "
          "'robotiq_2f85::right finger joint'.")

def set_gripper_q(q):
    if not (gripper_left and gripper_right):
        return
    q = float(np.clip(q, GRIP_MIN, GRIP_MAX))
    gripper_left.setPosition(q)
    gripper_right.setPosition(q)

def set_gripper_normalized(g):
    g = float(np.clip(g, 0.0, 1.0))
    q = GRIP_MIN + g * (GRIP_MAX - GRIP_MIN)
    set_gripper_q(q)

# -----------------------------------------------------------------------------
# Wrist camera (BGRA -> BGR)
# -----------------------------------------------------------------------------
cam = _safe_get_device("wrist_camera")
if cam:
    cam.enable(timestep)
    cam_w, cam_h = cam.getWidth(), cam.getHeight()
    print(f"[camera] Enabled wrist_camera at {cam_w}x{cam_h}")
else:
    cam_w = cam_h = None
    print("[camera] No device named 'wrist_camera' found.")

def get_camera_bgr():
    if not cam:
        return None
    img_bytes = cam.getImage()
    if img_bytes is None:
        return None
    bgra = np.frombuffer(img_bytes, dtype=np.uint8).reshape((cam_h, cam_w, 4))
    bgr = bgra[:, :, :3]  # drop alpha (BGRA -> BGR)
    return bgr

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
tt = 0
rpwv = RobotPlacerWithVision()

trial_active = False
trial_id = None

def command_initial_pose():
    # keep gripper open while waiting
    return list(initial_pose) + [0.0]

while robot.step(timestep) != -1:
    # receive commands
    while rx.getQueueLength() > 0:
        msg = json.loads(rx.getString())
        rx.nextPacket()

        if msg.get("cmd") == "start":
            trial_id = int(msg.get("trial_id", -1))
            illum = float(msg.get("illum", 0.0))
            seed = int(msg.get("seed", 0))
            target_color = msg.get('target_color')
            rpwv.start_trial(trial_id=trial_id, tt=tt, illum=illum, seed=seed, target_color=target_color)
            trial_active = True

    bgr = get_camera_bgr()
    current_q = [ps.getValue() for ps in sensors]  # real joint feedback

    if not trial_active:
        desired_command = command_initial_pose()
    else:
        desired_command = rpwv.getRobotCommand(tt, current_q, bgr)

    # apply arm command
    for j, motor in enumerate(motors):
        motor.setPosition(desired_command[j])

    # gripper: desired_command[-1] is 0(open) / 0.8(close)
    if desired_command[-1] > 0.4:
        set_gripper_normalized(1.0)
    else:
        set_gripper_normalized(0.0)

    # if episode done, send result once
    done_msg = rpwv.pop_done()
    if done_msg is not None:
        tx.send(json.dumps(done_msg).encode("utf-8"))
        trial_active = False
        trial_id = None

    tt += 1

if _HAS_CV2:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
