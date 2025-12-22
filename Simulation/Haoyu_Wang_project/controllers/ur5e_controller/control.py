"""Robot-side control logic (IK + vision-based FSM).

This file is imported by ur5e_controller.py.
"""

# you may find all of these packages helpful!
from kinematic_helpers import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import math
import lib
import detect
from typing import Optional, Dict, Any, Tuple
import time
import json
import os
from PIL import Image

experiment_2 = False

def forward_kinematics(q):
    T = T01(q[0]) @ T12(q[1]) @ T23(q[2]) @ T34(q[3]) @ T45(q[4]) @ T56(q[5])
    return T


def _compute_Ts(q):
    Ts = [np.eye(4)]
    fs = [T01, T12, T23, T34, T45, T56]
    Tcur = np.eye(4)
    for i, f in enumerate(fs):
        Tcur = Tcur @ f(q[i])
        Ts.append(Tcur.copy())
    return Ts  # Ts[0] = I, Ts[1] = T01, ..., Ts[6] = T06


def jacobian(q):
    Ts = _compute_Ts(q)
    p_e = Ts[-1][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        T0i = Ts[i]
        z = T0i[:3, 2]
        pi = T0i[:3, 3]
        J[:3, i] = np.cross(z, p_e - pi)
        J[3:, i] = z
    return J


def pose_error(Tcur, desired_pose):
    p_d = np.asarray(desired_pose[:3], dtype=float)
    rvec_d = np.asarray(desired_pose[3:6], dtype=float)
    R_d = R.from_rotvec(rvec_d).as_matrix()

    p = Tcur[:3, 3]
    R_c = Tcur[:3, :3]

    e_pos = p_d - p
    R_err = R_d @ R_c.T
    e_rot = R.from_matrix(R_err).as_rotvec()
    return np.hstack([e_pos, e_rot])


def inverse_kinematics(desired_pose, current_q,
                       max_iters=15, pos_tol=5e-5, rot_tol=2e-3,
                       damp=1e-2):
    q = np.asarray(current_q, dtype=float)

    for _ in range(max_iters):
        Tcur = forward_kinematics(q)
        e = pose_error(Tcur, desired_pose)
        if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
            break

        J = jacobian(q)

        JJt = J @ J.T
        A = JJt + (damp**2) * np.eye(6)
        JDLS = J.T @ np.linalg.inv(A)

        dq = JDLS @ e
        q = q + dq
    return q.tolist()


initial_pose = [0.15, -1.6, 1.5, -1.6, -1.70, 1.03]

def _to_uint8_rgb(bgr: np.ndarray) -> np.ndarray:
    """Webots gives BGR uint8; ensure we have RGB uint8."""
    if bgr is None:
        return None
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _detect_square_rgb(rgb_u8: np.ndarray, color: str, used_retinex: bool) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Detect red/blue square using HSV helper in test.py."""
    if rgb_u8 is None:
        return None, None
    if color == "red":
        if used_retinex:
            return detect.find_red_square_hsv(rgb_u8, input_space="RGB")
        else:
            return detect.normal_search(rgb_u8, color)
    elif color == "blue":
        if used_retinex:
            return detect.find_blue_square_hsv(rgb_u8, input_space="RGB")
        else:
            return detect.normal_search(rgb_u8, color)
    else:
        raise ValueError("color must be 'red' or 'blue'")


class RobotPlacerWithVision:


    """
    Multi-rate visual servo (Route A):

    - Only run Retinex when we are in a 'sense' moment (robot stopped).
    - After we get (cx, cy), we take one small Cartesian step (IK once),
    then we 'blind move' for a few ticks without doing vision.
    - Repeat until pixel error is small, then descend with the same multi-rate idea.
    """

    def __init__(self):
        self.state = 'start'

        # experimental parameters
        self.retinex_count = 0
        self.t_start = 0
        self.t_end = 0

        self.wall_t_start = None           # episode wall start
        self.wall_retinex_s = 0.0          # retinex wall time accumulator

        # vision/debug
        self.mask = None
        self.info = None
        self.color = None  # 'red' or 'blue'
        self.target_color = None  # 'red'/'blue' if forced by Supervisor

        # -------- multi-rate servo knobs --------
        self.pending_q = None
        self.pre_ex = None
        self.pre_ey = None
        self.pending_ticks = 0
        self.settle_q_tol = 2e-3   

        # descend logic
        self.descend_goal_z = 0.164
        self.descend_step_z_low = 0.008
        self.descend_step_z_medium = 0.05
        self.descend_step_z_high = 0.12

        # grip/drop
        self.grip_tick = 0
        self.drop_tick = 0

        # offsets 
        self.align_pix_err = 20
        self.step_size_align = 8e-4

        self.desc_tolerance = 8
        self.camera_offset_y = 10
        self.camera_offset_x = 160

        # safety
        self.lost_tick = 0
        self.lost_tolerance = 30
        
        self.retinex_used = False

        # batch-experiment bookkeeping
        self.trial_id = None
        self.illum = None
        self.seed = None
        self.done_flag = False
        self.done_msg = None

        self.enhanced = False
        self.red = False
        self.blue = False
    

    def start_trial(self, trial_id: int, tt: int, illum: float = None, seed: int = None, target_color: str = None):
        """Called by the robot-side controller when Supervisor sends a new start command."""
        self.trial_id = int(trial_id)
        self.illum = illum
        self.seed = seed


        self.target_color = target_color if target_color in ('red','blue') else None
        # reset FSM + counters
        self.state = "start"
        self.color = None
        self.mask = None
        self.info = None

        self.start_OK_count = 0
        self.lost_tick = 0
        self.pending_q = None
        self.pending_ticks = 0
        self.retinex_used = False

        self.retinex_count = 0
        self.t_start = tt
        self.t_end = tt

        self.wall_t_start = time.perf_counter()
        self.wall_retinex_s = 0.0

        self.done_flag = False
        self.done_msg = None

        self.fixed_rvec = None
        self._refresh()

    def pop_done(self):
        """Return the done message once (and clear it)."""
        if not self.done_flag:
            return None
        msg = self.done_msg
        self.done_flag = False
        self.done_msg = None
        return msg

    def _finish_trial(self, tt: int, success: int, reason: str, picked_color: str = None):
        """Mark current trial finished (success or failure) and prepare a message for Supervisor."""
        self.t_end = tt
        sim_time = float(self.t_end - self.t_start)
        wall_time = float(time.perf_counter() - self.wall_t_start) if self.wall_t_start is not None else None

        self.done_msg = {
            "event": "done",
            "trial_id": int(self.trial_id) if self.trial_id is not None else -1,
            "success": int(success),
            "reason": str(reason),
            "sim_time": sim_time,
            "wall_time": wall_time,
            "ret_calls": int(self.retinex_count),
            "ret_wall": float(self.wall_retinex_s),
            "picked_color": picked_color,
        }
        self.done_flag = True

    # ----------------------------
    # Vision: raw -> (optional) Retinex -> HSV detection
    # ----------------------------
    def _sense_color(self, bgr: np.ndarray, color: str, allow_retinex: bool = True):
        """Return (mask_full, info_full) in full-res pixel coords."""
        rgb = _to_uint8_rgb(bgr)
        if rgb is None:
            return None, None

        # 1) Try detection on raw image first (fast path)
        mask, info = _detect_square_rgb(rgb, color, used_retinex = False)
        if info is not None:
            self.mask, self.info = mask, info
            self.retinex_used = False
            return mask, info

        # 2) If raw fails, optionally run Retinex (slow path, but called sparsely)
        if not allow_retinex:
            self.mask, self.info = mask, None
            return mask, None

        if self.enhanced == True and self.state == 'search':
            R = self.pre_R
        else:
            img_np = rgb.astype(np.float32) / 255.0

            # This call is expensive (~6.5s in your report). In Webots, this will freeze the sim time (OK).
            t0 = time.perf_counter()
            enh_small_u8, R, L = lib.enhance_image(img_np)
            # print("enhance call")
            dt = time.perf_counter() - t0
            self.retinex_count += 1
            self.wall_retinex_s += dt
            self.pre_R = R
            self.enhanced = True

        # cv2.imshow("R",R)
        
        self.retinex_used = True

        mask_s, info_s = _detect_square_rgb(R, color, used_retinex=True)

        if info_s is None:
            self.mask, self.info = mask_s, None
            return mask_s, None

        cx, cy = info_s["center"]

        if "bbox_axis" in info_s:
            x, y, w, h = info_s["bbox_axis"]
        else:
            x, y, w, h = info_s.get("bbox", (0, 0, 0, 0))
        bbox = (int(x), int(y), int(w), int(h))

        info = dict(info_s)
        info["center"] = (cx, cy)
        info["bbox"] = bbox

        self.mask, self.info = mask_s, info
        return mask_s, info

    def _sense_any_color(self, bgr: np.ndarray, allow_retinex: bool = True):
        best = None
        best_color = None
        best_mask = None

        for c in ("red", "blue"):
            if c == "red" and self.red == False or c == "blue" and self.blue == False:
                mask, info = self._sense_color(bgr, c, allow_retinex=allow_retinex)
                if info is None:
                    continue
                area = float(info.get("area", 0.0))
                if (best is None) or (area > best.get("area", 0.0)):
                    best = info
                    best_color = c
                    best_mask = mask

        if best is None:
            self.mask, self.info = (best_mask, None)
            return None, None, None

        self.color = best_color
        self.mask = best_mask
        self.info = best
        return best_mask, best, best_color
    def _settled(self, current_q):
        if self.pending_q is None:
            return True
        dq = np.abs(np.array(current_q[:6]) - np.array(self.pending_q[:6]))
        return float(dq.max()) < self.settle_q_tol
    # ----------------------------
    # FSM states
    # ----------------------------
    def state_start(self, current_q):
        q_target = list(initial_pose)
        
        T_init = forward_kinematics(initial_pose) 
        self.fixed_rvec = R.from_matrix(T_init[:3, :3]).as_rotvec()

        current_arr = np.array(current_q[:6])
        target_arr = np.array(initial_pose)
        
        error = np.linalg.norm(current_arr - target_arr)
        
        if error < 0.05:
            self.start_OK_count += 1
        else:
            self.start_OK_count = 0  
            
        if self.start_OK_count >= 8:
            status = 1
        else:
            status = 0
            
        return q_target + [0.0], int(status)

    def state_search(self, current_q, current_image_bgr):
        """Sense once (possibly Retinex) to decide which color cube is visible."""
        if self.target_color in ("red", "blue"):
            mask, info = self._sense_color(current_image_bgr, self.target_color, allow_retinex=True)
            self.mask, self.info = mask, info
            self.color = self.target_color if info is not None else None
        else:
            self.mask, self.info, self.color = self._sense_any_color(current_image_bgr, allow_retinex=True)
        if self.info is None:
            # hold position
            return list(current_q[:6]) + [0.0], 0
        return list(current_q[:6]) + [0.0], 1

    def state_align_multirate(self, current_q, current_image_bgr, lost = False):
        """Align end-effector above the cube with multi-rate sense/move."""

        if self.pending_q is not None and (not self._settled(current_q)):
            self.pending_ticks += 1
            print("pending:", self.pending_ticks)
            return self.pending_q, 0
        self.pending_ticks = 0
        self.pending_q = None
        
        Tcur = forward_kinematics(current_q)
        position = Tcur[:3, 3].copy()

        _, info = self._sense_color(current_image_bgr, self.color, allow_retinex=True)
        if info is None:
            if lost == True and (self.pre_ex is not None and self.pre_ey is not None):
                # do one step with previous calculation
                ex = self.pre_ex * 0.6
                ey = self.pre_ey * 0.6
            else:
                return list(current_q[:6] + [0.0]), -1
        else:

            # 2) sense phase: run detection (raw or Retinex) ONCE


            cx, cy = info["center"]
            grip_cx = current_image_bgr.shape[1] / 2 + self.camera_offset_x
            grip_cy = current_image_bgr.shape[0] / 2 + self.camera_offset_y

            ex = cx - grip_cx
            ey = cy - grip_cy

            print("pres:",self.pre_ex, self.pre_ey)
            if abs(ex) <= self.align_pix_err and abs(ey) <= self.align_pix_err:
                return list(current_q[:6] + [0.0]), 1


        z = float(position[2]) if abs(position[2]) > 1e-6 else 1.0

        dx = float(ex) * self.step_size_align * z
        dy = float(ey) * self.step_size_align * z
        
        dx = float(np.clip(dx, -0.015, 0.015))
        dy = float(np.clip(dy, -0.015, 0.015))


        position = position + np.array([-dx, +dy, 0.0])
        rvec = self.fixed_rvec
        desired_pose = [position[0], position[1], position[2], rvec[0], rvec[1], rvec[2]]
        q6 = inverse_kinematics(desired_pose, current_q)


        print("dx,", dx," dy ", dy, "ex,", ex, "ey", ey, "cx", cx, "cy", cy, "grab_cx", grip_cx, "grab_cy", grip_cx)
        q = list(q6)
        if len(q) < 7:
            q.append(0.0)
        else:
            q[6] = 0.0
        
        self.pre_ex = ex
        self.pre_ey = ey
        self.pending_q = q
        self.pending_ticks = 0
        return q, 0

    def state_descend_multirate(self, current_q, current_image_bgr, lost):
        
        Tcur = forward_kinematics(current_q)
        position = Tcur[:3, 3].copy()
        rvec = self.fixed_rvec

        # 1) blind-move phase        
        if self.pending_q is not None and (not self._settled(current_q)):
            self.pending_ticks += 1
            print("pending:", self.pending_ticks)
            return self.pending_q, 0
        self.pending_ticks = 0
        self.pending_q = None

        # 2) sense phase
        _, info = self._sense_color(current_image_bgr, self.color, allow_retinex=True)
        if info is None:
            if lost == True:
                print("blind lost")
                # do one step with previous calculation
                ex = self.pre_ex * 0.3
                ey = self.pre_ey * 0.3
            else:
                if position[2] < self.descend_goal_z:
                    return list(current_q[:6] + [0.0]), 1
                if (position[2] - self.descend_goal_z) < 0.01:
                    print("one last step")
                    desired_pose = [position[0], position[1], position[2] - 0.002, rvec[0], rvec[1], rvec[2]]
                    q = inverse_kinematics(desired_pose, current_q)
                    if len(q) < 7:
                        q.append(0.0)
                    else:
                        q[6] = 0.0
                    self.pending_q = q
                    self.pending_ticks = 0
                    return q, 1
                else:
                    return list(current_q[:6] + [0.0]), -1
        else:
            cx, cy = info["center"]
            grip_cx = current_image_bgr.shape[1] / 2 + self.camera_offset_x
            grip_cy = current_image_bgr.shape[0] / 2 + self.camera_offset_y
            ex, ey = (cx - grip_cx), (cy - grip_cy)

        # 2a) first keep XY aligned
        if max(abs(ex), abs(ey)) > self.desc_tolerance:
            dz = position[2] - self.descend_goal_z
            
            step_xy = 0.0004 + (dz - self.descend_goal_z) * 0.002 
            print("step_xy calculated:", step_xy)
            if dz <= 0.045:
                step_xy = np.clip(step_xy, 0.00012, 0.001)
            else:
                step_xy = np.clip(step_xy, 0.0004, 0.001)

            print("step_xy using:", step_xy)
            dx = float(np.clip(ex * step_xy, -0.008, 0.008))
            dy = float(np.clip(ey * step_xy, -0.008, 0.008))
            position[0] += -dx
            position[1] +=  dy
            done = 0 
            
            print("Alingnning step dx, dy, ex, ey, dz", dx, dy , ex , ey, dz)
        else:
            # 2b) if aligned, go down a fixed step each sense
            dz = position[2] - self.descend_goal_z
            print("GO DOWNNNNNNN", dz)
            if dz <= 0.07:
                step_z = self.descend_step_z_low
            elif dz <= 0.27: 
                step_z = self.descend_step_z_medium
            else:
                step_z = self.descend_step_z_high
            print("step_z using:", step_z)

            step_y = step_z * 0.2
            if position[2] > self.descend_goal_z + 0.001:
                position[2] = position[2] - step_z
                position[1] -= step_y
                print("calculated position z", position[2])
                done = 0
            else:
                done = 1

        desired_pose = [position[0], position[1], position[2], rvec[0], rvec[1], rvec[2]]
        q6 = inverse_kinematics(desired_pose, current_q)

        q = list(q6)
        if len(q) < 7:
            q.append(0.0)
        else:
            q[6] = 0.0

        self.pending_q = q
        self.pending_ticks = 0
        return q, done

    def state_lift(self, current_q, goal=0.6, step_size=0.003):
        """Simple lift with fixed step size (no vision)."""
        Tcur = forward_kinematics(current_q)
        position = Tcur[:3, 3].copy()
        rvec = R.from_matrix(Tcur[:3, :3]).as_rotvec()

        if position[2] + step_size < goal:
            position[2] += step_size
            done = 0
        else:
            position[2] = goal
            done = 1

        desired_pose = [position[0], position[1], position[2], rvec[0], rvec[1], rvec[2]]
        q6 = inverse_kinematics(desired_pose, current_q)

        q = list(q6)
        if len(q) < 7:
            q.append(1)
        else:
            q[6] = 1
        return q, done

    def state_move(self, current_q, target_red=[0.409, 0.636], target_blue = [0.453, -0.455], step_size=0.002, err=0.0005):
        """Move to bin (no vision)."""

        if self.color == 'red':
            target = target_red
        else:
            target = target_blue            
        Tcur = forward_kinematics(current_q)
        position = Tcur[:3, 3].copy()
        rvec = R.from_matrix(Tcur[:3, :3]).as_rotvec()

        e_x = target[0] - position[0]
        e_y = target[1] - position[1]
        e_dis = math.hypot(e_x, e_y)

        if e_dis < err:
            done = 1
            new_pos = position
        else:
            done = 0
            new_pos = position.copy()
            new_pos[0] += e_x / e_dis * step_size
            new_pos[1] += e_y / e_dis * step_size

        desired_pose = [new_pos[0], new_pos[1], new_pos[2], rvec[0], rvec[1], rvec[2]]
        q6 = inverse_kinematics(desired_pose, current_q)

        q = list(q6)
        if len(q) < 7:
            q.append(0.8)
        else:
            q[6] = 0.8
        return q, done

    def _handle_lost(self, tt: int):
        # print("lost tick")
        self.lost_tick += 1
        if self.lost_tick >= self.lost_tolerance:
            # fail current trial and report to Supervisor
            self._finish_trial(tt=tt, success=0, reason="lost", picked_color=self.color)
            self.t_start = tt
            # reset internal FSM
            self.state = "start"
            self.lost_tick = 0
            self.pending_q = None
        else:
            # try previous calculation
            pass
            

    def _refresh(self):
        self.start_OK_count = 0

        self.mask = None
        self.info = None
        self.color = None  # 'red' or 'blue'
        self.target_color = None  # 'red'/'blue' if forced by Supervisor

        self.pending_q = None
        self.pre_ex = None
        self.pre_ey = None
        self.pending_ticks = 0
        self.settle_q_tol = 2e-3   

        self.grip_tick = 0
        self.drop_tick = 0

        self.lost_tick = 0
        self.lost_tolerance = 30
        
        self.retinex_used = False
        self.enhanced = False

    def getRobotCommand(self, tt, current_q, current_image_bgr):
        # if self.color is not None:
        #     print(self.color)
        print(self.state)

        Tcur = forward_kinematics(current_q)
        position = Tcur[:3, 3].copy()
        # print(position)
        print(current_q)

        # cv2.imshow("img", current_image_bgr)

        # if self.mask is not None:
        #     try:
        #         x, y, w, h = self.info["bbox_axis"]
        #         vis = self.mask.copy()
        #         cv2.rectangle(vis, (x,y), (x+w, y+h), 128, 2)  # 灰色框更显眼
        #         cv2.imshow("mask", vis)
        #         cv2.waitKey(1)
        #     except Exception:
        #         pass

        q = list(current_q[:6])  # ensure list

        match self.state:
            case 'start':
                cmd, status = self.state_start(current_q)
                if status == 1:
                    self.state = 'search'
                    self.start_OK_count = 0
                    if self.color == 'blue':
                        self.blue = True
                    elif self.color == 'red':
                        self.red = True
                return cmd

            case 'search':
                cmd, found = self.state_search(current_q, current_image_bgr)
                if found:
                    self.pre_R = None
                    self.enhanced = False
                    self.pre_ex = None
                    self.pre_ey = None
                    self.lost_tick = 0
                    self.pending_q = None
                    self.pending_ticks = 0
                    self.state = 'align'
                    wall_total = time.perf_counter() - self.wall_t_start if self.wall_t_start is not None else None

                    # print("search Retine %i", self.retinex_count)
                    self.t_end = tt
                    total_time = self.t_end - self.t_start
                    wall_total = time.perf_counter() - self.wall_t_start if self.wall_t_start is not None else None

                    print(f"TOTAL sim={total_time}  wall={wall_total:.3f}s  "
                        f"ret_calls={self.retinex_count}  ret_wall={self.wall_retinex_s:.3f}s")

                return cmd

            case 'align':
                if self.lost_tick == 0:
                    lost = False
                else:
                    lost = True
                cmd, status = self.state_align_multirate(current_q, current_image_bgr, lost)
                if status == 1:
                    self.pre_ex = 0
                    self.pre_ey = 0
                    self.lost_tick = 0
                    self.pending_q = None
                    self.pending_ticks = 0
                    self.state = 'descend'
                    # print("align Retine %i", self.retinex_count)
                    self.t_end = tt
                    wall_total = time.perf_counter() - self.wall_t_start if self.wall_t_start is not None else None
                    total_time = self.t_end - self.t_start
                    print(f"TOTAL sim={total_time}  wall={wall_total:.3f}s  "
                        f"ret_calls={self.retinex_count}  ret_wall={self.wall_retinex_s:.3f}s")
                elif status == -1:
                    self._handle_lost(tt)
                return cmd

            case 'descend':
                if self.lost_tick == 0:
                    lost = False
                else:
                    lost = True
                cmd, status = self.state_descend_multirate(current_q, current_image_bgr, lost)
                if status == 1:
                    self.pre_ex = 0
                    self.pre_ey = 0
                    self.lost_tick = 0
                    self.pending_q = None
                    self.pending_ticks = 0
                    self.state = 'close'
                    # print("descend Retine %i", self.retinex_count)
                    self.t_end = tt
                    total_time = self.t_end - self.t_start
                    # print("descend time: %d", total_time)
                    wall_total = time.perf_counter() - self.wall_t_start if self.wall_t_start is not None else None
                    total_time = self.t_end - self.t_start
                    print(f"TOTAL sim={total_time}  wall={wall_total:.3f}s  "
                        f"ret_calls={self.retinex_count}  ret_wall={self.wall_retinex_s:.3f}s")
                elif status == -1:
                        self._handle_lost(tt)
                return cmd

            case 'close':
                # close gripper for a while
                if len(q) < 7:
                    q.append(0.8)
                else:
                    q[6] = 0.8
                self.grip_tick += 1
                if self.grip_tick > 150:
                    self.grip_tick = 0
                    self.state = 'lift'
                return q

            case 'lift':
                cmd, done = self.state_lift(current_q)
                if done:
                    self.state = 'move'
                return cmd

            case 'move':
                cmd, done = self.state_move(current_q)
                if done:
                    self.state = 'drop'
                return cmd

            case 'drop':
                if len(q) < 7:
                    q.append(0.0)
                else:
                    q[6] = 0.0
                self.drop_tick += 1
                if self.drop_tick > 80:
                    self.drop_tick = 0
                    picked = self.color
                    self._finish_trial(tt=tt, success=1, reason="success", picked_color=picked)
                    self.t_start = tt
                    # reset internal state (robot-side loop will go back to waiting)
                    self.color = None
                    self._refresh
                    self.state = 'start'
                return q

        # fallback
        if len(q) < 7:
            q.append(0.0)
        return q