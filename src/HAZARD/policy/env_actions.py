import sys
import os
import time
import cv2
from src.HAZARD.policy.dstar.d_star_lite import get_dstar_weight, DStarLite
from src.HAZARD.policy.dstar.grid import OccupancyGridMap
from openai import OpenAI
import numpy as np
import json
import re
import random
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)
sys.path.append(PATH)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import json
import re
import random

from typing import List, Union, Optional
import numpy as np
from tdw.replicant.ik_plans.ik_plan_type import IkPlanType
from tdw.replicant.action_status import ActionStatus
from src.HAZARD.policy.astar import get_astar_path, get_astar_weight
from tdw.tdw_utils import TDWUtils
from tdw.replicant.arm import Arm

"""
All actions here are single-agent
"""

def visualize_obs(env, obs, suffix="0", save_dir=os.path.join(PATH, "logs"), astar_path=None):
    if obs is None:
        obs = env.controller._obs()
    sem_map = obs["sem_map"]
    rgb = obs["raw"]["rgb"]
    rgb_seg = obs["raw"]["seg_mask"]
    """
    output an image of the semantic map
    red: agent
    white: unexplored
    grey: explored and unoccupied
    black: occupied
    """
    explored = sem_map["explored"]
    height = sem_map["height"]
    ID = sem_map["id"]
    w, h = explored.shape
    img = np.zeros((w, h, 3), dtype=np.uint8)
    obj = np.zeros((w, h, 3), dtype=np.uint8)
    obj[ID > 0.5] = [0, 0, 0]
    obj[ID < 0.5] = [255, 255, 255]
    height_img = np.copy(height)
    height_img.astype(np.uint8)
    height_img = np.expand_dims(height_img, axis=2)
    height_img = np.repeat(height_img, 3, axis=2)
    height_img *= int(255 / height.max())
    height_img[height <= 0] = [0, 0, 0]
    height_img[height > 1.3]=[255,0,0]
    img[explored == 0] = [255, 255, 255] # this color is white
    img[explored == 1] = [200, 200, 200] # this color is grey
    img = img - np.reshape(200 * np.minimum(height / 2, 1.0) * (height > 0.3), (w, h, 1))
    pos = env.controller.agents[0].dynamic.transform.position
    offset = env.controller.sem_map.map_offset
    grid_size = env.controller.sem_map.grid_size
    pos = [int(pos[0] // grid_size + offset[0]), int(pos[2] // grid_size + offset[1])]
    # set nearby pixels to red
    for i in range(-1, 2):
        for j in range(-1, 2):
            img[pos[0] + i, pos[1] + j] = np.array([255, 0, 0])
    # mark path
    if astar_path is not None:
        for p in astar_path:
            img[p[0], p[1]] = np.array([0, 255, 0])
    
    print(height.max())
    # save the image
    print("rgb", rgb.shape)
    # rgb to bgr
    rgb = rgb[[2, 1, 0], :, :]
    img = img[:, :, [2, 1, 0]]
    obj = obj[:, :, [2, 1, 0]]
    height_img = height_img[:, :, [2, 1, 0]]
    explored = np.copy(explored)
    explored *= 255
    explored.astype(np.uint8)
    explored = np.expand_dims(explored, axis=2)
    explored = np.repeat(explored, 3, axis=2)
    explored = explored[:, :, [2, 1, 0]]
    cv2.imwrite(os.path.join(save_dir, f"sem_map_{suffix}.png"), img)
    cv2.imwrite(os.path.join(save_dir, f"rgb_{suffix}.png"), rgb.transpose(1, 2, 0) * 255)
    cv2.imwrite(os.path.join(save_dir, f"obj_map_{suffix}.png"), obj)
    cv2.imwrite(os.path.join(save_dir, f"height_map_{suffix}.png"), height_img)
    cv2.imwrite(os.path.join(save_dir, f"explore_{suffix}.png"), explored)
    cv2.imwrite(os.path.join(save_dir, f"seg_mask_{suffix}.png"), rgb_seg)

# --- small helpers to be robust ---
def _xz(v):
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size >= 3:
        return np.array([a[0], a[2]], dtype=float)
    elif a.size == 2:
        # assume (x, z)
        return a.astype(float)
    else:
        raise ValueError(f"Bad position size {a.size}: {v}")

def _vec3(v, fallback_y=0.0):
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size >= 3:
        return [float(a[0]), float(a[1]), float(a[2])]
    elif a.size == 2:
        # (x, z) -> (x, y, z)
        return [float(a[0]), float(fallback_y), float(a[1])]
    else:
        raise ValueError(f"Bad position size {a.size}: {v}")

def save_path_map(env,
                  obs=None,
                  save_dir=os.path.join(PATH, "logs"),
                  astar_path=None,
                  dstar_path=None,  # Add D* path parameter
                  traj_color_bgr=(255, 0, 0),     # blue line (BGR)
                  astar_color_bgr=(0, 255, 0),    # green line for A* path
                  dstar_color_bgr=(0, 0, 255),    # red line for D* path
                  thickness=2):
    """
    Draws paths on semantic map:
    - Agent trajectory in blue
    - A* path in green
    - D* path in red
    """
    save_dir=os.path.join(save_dir, '0')
    # 0) Get obs + sem map
    if obs is None:
        obs = env.controller._obs()
    sem_map = obs["sem_map"]
    explored = sem_map["explored"]
    height = sem_map["height"]
    ID = sem_map["id"]
    w, h = explored.shape

    os.makedirs(save_dir, exist_ok=True)

    # 1) Base semantic map image (same palette as your visualize_obs)
    img = np.zeros((w, h, 3), dtype=np.uint8)
    img[explored == 0] = [255, 255, 255]   # white = unexplored
    img[explored == 1] = [200, 200, 200]   # grey = explored
    img = img - np.reshape(200 * np.minimum(height / 2, 1.0) * (height > 0.3), (w, h, 1))

    # 2) Current agent grid position (row,col)
    pos_real = np.asarray(env.controller.agents[0].dynamic.transform.position, dtype=float)
    offset = env.controller.sem_map.map_offset
    grid_size = env.controller.sem_map.grid_size
    pos_grid = [int(pos_real[0] // grid_size + offset[0]), int(pos_real[2] // grid_size + offset[1])]

    # 3) Accumulate trajectory in env.controller._traj_history (list of [row,col])
    if not hasattr(env.controller, "_traj_history"):
        env.controller._traj_history = []
    # append if new & in-bounds
    r, c = pos_grid
    if 0 <= r < w and 0 <= c < h:
        if (len(env.controller._traj_history) == 0) or (env.controller._traj_history[-1] != [r, c]):
            env.controller._traj_history.append([r, c])

    # 4) Convert to BGR for OpenCV drawing (polylines expects (x=col, y=row))
    bgr = img[:, :, [2, 1, 0]].copy()

    # 5) Draw entire trajectory as a polyline
    if len(env.controller._traj_history) >= 2:
        pts = np.array([[p[1], p[0]] for p in env.controller._traj_history], dtype=np.int32)  # (col,row)
        cv2.polylines(bgr, [pts], isClosed=False, color=traj_color_bgr, thickness=thickness)

    # 6) Overlay A* path if provided
    if astar_path is not None and len(astar_path) >= 2:
        astar_pts = []
        for r0, c0 in astar_path:
            if 0 <= r0 < w and 0 <= c0 < h:
                astar_pts.append([int(c0), int(r0)])
        if len(astar_pts) >= 2:
            cv2.polylines(bgr, [np.array(astar_pts, dtype=np.int32)], isClosed=False,
                         color=astar_color_bgr, thickness=max(1, thickness - 1))

    # 6.5) Overlay D* path if provided
    if dstar_path is not None and len(dstar_path) >= 2:
        dstar_pts = []
        for r0, c0 in dstar_path:
            if 0 <= r0 < w and 0 <= c0 < h:
                dstar_pts.append([int(c0), int(r0)])
        if len(dstar_pts) >= 2:
            cv2.polylines(bgr, [np.array(dstar_pts, dtype=np.int32)], isClosed=False,
                         color=dstar_color_bgr, thickness=max(1, thickness - 1))

    # 7) Mark current agent cell (small red square)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            rr, cc = r + di, c + dj
            if 0 <= rr < w and 0 <= cc < h:
                bgr[rr, cc] = (0, 0, 255)  # red in BGR

    # 8) Save
    print("Saving path map...")
    frame = env.controller.frame_count
    out_path = os.path.join(save_dir, f"path_{frame}.png")
    cv2.imwrite(out_path, bgr)
    return out_path
 
# def agent_walk_to(env: WindEnv, target: Union[int, np.ndarray, List], max_steps=100, reset_arms: bool = False, arrived_at=1.0):
def agent_walk_to(env, target: Union[int, np.ndarray, List], max_steps=100, reset_arms: bool = False, arrived_at=1.0,
                task=None, effect_on_agents=False, record_mode=False, use_dstar=False,feedback=False, previous_seen=None, target_object_id=None, last_action_info=None,):
    """Walk to target using either A* or D* path planning
    
    Returns:
        Tuple[bool, str, Tuple[float, float]]: (success, message, (distance_meters, distance_grid))
    """
    start_frame = env.controller.frame_count
    total_meters = 0.0
    total_grid = 0.0
    last_pos = env.controller.agents[0].dynamic.transform.position.copy()
    grid_size = env.controller.sem_map.grid_size
    if feedback:
        feedback_des = True
    else:
        s_time = time.time()
        feedback_des, action, previous_seen = llm_feedback(env, target_object_id,last_action_info, previous_seen,task=task)
        print("feedback_des, action, previous_seen", feedback_des, action, previous_seen)
        f_time = time.time()
        elapsed_time = f_time - s_time
        frame = elapsed_time
        
    if feedback_des == False:
        return 'feedback' , action, total_grid, previous_seen

    while True:
        current_pos = env.controller.agents[0].dynamic.transform.position
        
        # Convert TDW units (1 meter = 100 units) to meters
        step_meters = np.linalg.norm(current_pos[[0, 2]] - last_pos[[0, 2]]) / 100.0
        
        # Calculate grid cell distance
        last_grid = env.controller.real_to_grid(last_pos)
        current_grid = env.controller.real_to_grid(current_pos)
        step_grid = np.linalg.norm(np.array(current_grid) - np.array(last_grid))
        
        
        total_meters += step_meters
        total_grid += step_grid
        last_pos = current_pos.copy()

        target_pos = env.controller.manager.objects[target].position if isinstance(target, int) else np.array(target)
        
        if np.linalg.norm(current_pos[[0, 2]] - target_pos[[0, 2]]) < arrived_at:
            env.controller.agents[0].collision_detection.avoid = True
            env.controller.agents[0].collision_detection.objects = True
            return True, "success", total_grid, previous_seen
        
        if env.controller.frame_count - start_frame > max_steps:
            env.controller.agents[0].collision_detection.avoid = True
            env.controller.agents[0].collision_detection.objects = True
            return False, "max steps reached", total_grid, previous_seen
        
        agent_pos = env.controller.real_to_grid(current_pos)
        target_pos = env.controller.real_to_grid(target_pos)
        obs = env.controller._obs()
        sem_map = obs["sem_map"]
        
        if isinstance(target, int) and not np.any(sem_map["id"] == env.controller.manager.id_renumbering[target]):
            env.controller.agents[0].collision_detection.avoid = True
            env.controller.agents[0].collision_detection.objects = True
            return False, "target not in vision or memory", total_grid, previous_seen

        # Get path using either D* or A*
        if use_dstar:
            # Get D* path
            weight = get_dstar_weight(sem_map=sem_map, origin=agent_pos, destination=target_pos)
            
            
            occ_map = OccupancyGridMap(x_dim=weight.shape[0], 
                                      y_dim=weight.shape[1],
                                      exploration_setting='8N')
            occ_map.grid = np.where(weight > 1000, 255, 0).astype(np.uint8)
            
            if not hasattr(env.controller, 'dstar') or env.controller.dstar.s_goal != tuple(target_pos):
                env.controller.dstar = DStarLite(occ_map, 
                                               s_start=tuple(agent_pos),
                                               s_goal=tuple(target_pos))
            
            dstar_path, _, _ = env.controller.dstar.move_and_replan(tuple(agent_pos))
            
            # Also get A* path for comparison
            astar_weight = get_astar_weight(sem_map=sem_map, origin=agent_pos, destination=target_pos)
            astar_path = get_astar_path(weight=astar_weight, origin=agent_pos, destination=target_pos)
            
            # Save both paths
            # save_path_map(env, obs=obs, astar_path=astar_path, dstar_path=dstar_path)
            
            path = dstar_path if dstar_path is not None and len(dstar_path) > 1 else astar_path
        else:
            # Only A* path
            weight = get_astar_weight(sem_map=sem_map, origin=agent_pos, destination=target_pos)
            path = get_astar_path(weight=weight, origin=agent_pos, destination=target_pos)
            # save_path_map(env, obs=obs, astar_path=path)
        
        if path is None or len(path) <= 1:
            env.controller.agents[0].collision_detection.avoid = True
            env.controller.agents[0].collision_detection.objects = True
            return False, "don't know, maybe out of bounds", total_grid, previous_seen

        # Move to next point in path
        next_point = path[1]  # D* returns path as list of tuples
        env.controller.do_action(
            agent_idx=0,
            action="move_to",
            params={
                "target": TDWUtils.array_to_vector3(env.controller.grid_to_real(next_point)),
                "reset_arms": reset_arms,
                "arrived_at": 0.05 if record_mode else 0.5
            }
        )

        if not effect_on_agents or task == "fire":
            env.controller.next_key_frame(force_direction=None)
        elif task == "wind":
            wind_v = env.controller.manager.wind_v
            env.controller.next_key_frame(force_direction=wind_v)
        else:
            assert env.controller.manager.flood_manager.source_from == 'x_max'
            env.controller.next_key_frame(force_direction=np.array([-1, 0, 0]))

def low_level_action(env, action, effect_on_agents=False, task=None, **kwargs):
    assert action in ['move_by', 'turn_by', 'turn_to', 'reach_for']
    forbidden_params = ['max_distance', 'duration', 'scale_duration', 'arrived_at']
    for param in forbidden_params:
        assert param not in kwargs
    if 'target' in kwargs and type(kwargs['target']) == int:
        kwargs['target'] = env.controller.manager.id_renumbering[kwargs['target']]
    getattr(env.controller.agents[0], action)(**kwargs)
    if action != "move_by":
        effect_on_agents = False
    if not effect_on_agents or task == "fire":
        env.controller.next_key_frame()
    elif task == "wind":
        wind_v = env.controller.manager.wind_v
        env.controller.next_key_frame(force_direction=wind_v)
    else:
        assert env.controller.manager.flood_manager.source_from == 'x_max'
        env.controller.next_key_frame(force_direction=np.array([-1, 0, 0]))
    return True, "success"


def agent_walk_to_single_step(env, target: Union[int, np.ndarray, List], reset_arms: bool = False, arrived_at=1.0,
                              effect_on_agents=False, task=None, record_mode=False):
    agent_pos = env.controller.agents[0].dynamic.transform.position
    target_pos = env.controller.manager.objects[target].position if isinstance(target, int) else np.array(target)
    # print(target, env.controller.manager.objects[target].position, type(target))
    if np.linalg.norm(agent_pos - target_pos) < arrived_at:
        return True, "success"
    agent_pos = env.controller.real_to_grid(agent_pos)
    target_pos = env.controller.real_to_grid(target_pos)
    obs = env.controller._obs()
    sem_map = obs["sem_map"]
    if isinstance(target, int) and not np.any(sem_map["id"] == env.controller.manager.get_renumbered_id(target)):
        return False, "target not in vision or memory"
    
    weight = get_astar_weight(sem_map=sem_map, origin=agent_pos, destination=target_pos)
    path = get_astar_path(weight=weight, origin=agent_pos, destination=target_pos)
    # walk to first point
    if path is None or len(path) <= 1:
        return False, "don't know, maybe out of bounds"
    env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(path[1])),
                                                                    "reset_arms": reset_arms,
                                                                    "arrived_at": 0.05 if record_mode else 0.5})

    if not effect_on_agents or task == "fire":
        env.controller.next_key_frame()
    elif task == "wind":
        wind_v = env.controller.manager.wind_v
        env.controller.next_key_frame(force_direction=wind_v)
    else:
        assert env.controller.manager.flood_manager.source_from == 'x_max'
        env.controller.next_key_frame(force_direction=np.array([-1, 0, 0]))
    return True, "ongoing"

"""this action is environment independent"""
def agent_pickup(env, target: int, env_type: str = "what?"):
    if target not in env.controller.target_ids:
        return False, f"can only pick up target objects"

    arm = [Arm.left, Arm.right] if env_type == "wind" else Arm.right
    
    above_pos = env.controller.manager.objects[target].top()
    above_pos[1] += 100.0
    target_pos = env.controller.manager.objects[target].center()
    
    env.controller.do_action(agent_idx=0, action="turn_to", params={"target": target})
    status = env.controller.next_key_frame()[0][0]
    if status != ActionStatus.success:
        print("failed to turn to target")
        input()

    env.controller.do_action(agent_idx=0, action="reach_for", params={"target": TDWUtils.array_to_vector3(target_pos),
                                                                        "arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right,
                                                                        "arrived_at": 0.02,
                                                                        "max_distance": 2.5,
                                                                        "plan": IkPlanType.vertical_horizontal})
    status = env.controller.next_key_frame()[0][0]
    reach_success = (status == ActionStatus.success)

    # if not reach_success:
    #     env.controller.do_action(agent_idx=0, action="reset_arm",
    #                              params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
    #     env.controller.next_key_frame()
    #     env.controller.do_action(agent_idx=0, action="turn_by", params={"angle": random.randint(30, 330)})
    #     env.controller.next_key_frame()
    #     return False, f"cannot reach for this object, maybe too far"

    env.controller.communicate([{"$type": "set_kinematic_state", "id": target, "is_kinematic": False, "use_gravity": False}])
    if target in env.controller.other_containers:
        env.controller.communicate(
            {"$type": "teleport_object", "id": target, "position": TDWUtils.array_to_vector3(above_pos)})

    env.controller.do_action(agent_idx=0, action="grasp", params={"target": target, 
                                                                    "arm": Arm.right,
                                                                    "angle": None,
                                                                    "axis": None,})
    status = env.controller.next_key_frame()[0][0]
    
    env.controller.do_action(agent_idx=0, action="reset_arm", params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
    env.controller.next_key_frame()
    
    if status == ActionStatus.success:
        return True, f"success, reach for {'success' if reach_success else 'failed, may cause problems'}"
    elif status == ActionStatus.cannot_grasp:
        return False, "cannot grasp, maybe too far"
    else:
        return False, "failed to grasp"

"""turn around to find new targets"""
def agent_explore(env):
    TURN_TIMES = 12
    for i in range(TURN_TIMES):
        env.controller.do_action(agent_idx=0, action="turn_by", params={"angle": 360.0 / TURN_TIMES})
        status = env.controller.next_key_frame()[0][0]
        env.controller._obs()
        if status != ActionStatus.success:
            return False, 'can not turn around at this time'
    return True, 'success'

def agent_drop(env, container: Optional[int]=None, env_type: str = "what?"):
    arm = Arm.right
    try:
        grasp_id = env.controller.agents[0].dynamic.held_objects[arm]
    except:
        return False, "not holding an object"

    if hasattr(env.controller, "container_id"):
        assert Arm.left in env.controller.agents[0].dynamic.held_objects and env.controller.agents[0].dynamic.held_objects[Arm.left] == env.controller.container_id
        container = env.controller.container_id

    if (env_type == "flood" or env_type == "fire"):
        top = env.controller.manager.objects[container].top()
        top = top + np.array([0, 0.2, 0])

        env.controller.do_action(agent_idx=0, action="turn_to", params={"target": TDWUtils.array_to_vector3(top)})
        status = env.controller.next_key_frame()[0][0]
        if status != ActionStatus.success:
            print("failed to turn to target")
            input()

        top = env.controller.manager.objects[container].top()
        above = top + np.array([0, 0.2, 0])

        env.controller.do_action(agent_idx=0, action="reach_for", params={"target": TDWUtils.array_to_vector3(above),
                                                                        "absolute": True,
                                                                        "offhand_follows": False,
                                                                        "arm": Arm.left,
                                                                        "from_held": True,
                                                                        "held_point": "top",
                                                                        "max_distance": 1.0,
                                                                        "arrived_at": 0.05,
                                                                        "plan": IkPlanType.vertical_horizontal})
        status = env.controller.next_key_frame()[0][0]
        env.controller.do_action(agent_idx=0, action="reach_for", params={"target": TDWUtils.array_to_vector3(top),
                                                                        "absolute": True,
                                                                        "offhand_follows": False,
                                                                        "arm": Arm.right,
                                                                        "from_held": True,
                                                                        "held_point": "top",
                                                                        "max_distance": 1.0,
                                                                        "arrived_at": 0.05,
                                                                        "plan": IkPlanType.vertical_horizontal})
        status = env.controller.next_key_frame()[0][0]
        
        env.controller.do_action(agent_idx=0, action="drop", params={"arm": arm, "max_num_frames": 20})
        status = env.controller.next_key_frame()[0][0]
        # env.controller.communicate([{"$type": "destroy_object", "id": grasp_id}])
        # instead of destroying it, teleport it to [100, 0, 0] and make it kinematic
        destroy_commands = []
        destroy_commands.append({"$type": "teleport_object", "id": grasp_id, "position": {"x": 100, "y": 20, "z": 0}})
        destroy_commands.append({"$type": "set_kinematic_state", "id": grasp_id, "is_kinematic": True, "use_gravity": False})
        if hasattr(env.controller, "finished") and grasp_id not in env.controller.finished:
            env.controller.finished.append(grasp_id)
        if env_type == "fire":
            env.controller.manager.objects[grasp_id].temperature_threshold = 4000
        env.controller.communicate(destroy_commands)
        # env.controller.do_action(agent_idx=0, action="reset_arm",
        #                          params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
        env.controller.do_action(agent_idx=0, action="reset_arm", params={"arm": [Arm.left, Arm.right]})
        env.controller.next_key_frame()
        return True, "successfully drop and put away"

    if container is None:
        env.controller.do_action(agent_idx=0, action="drop", params={"arm": arm, "max_num_frames": 20})
        status = env.controller.next_key_frame()[0][0]
        env.controller.do_action(agent_idx=0, action="reset_arm",
                                 params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
        env.controller.next_key_frame()
        if status == ActionStatus.success:
            return True, "drop success"
        elif status == ActionStatus.still_dropping:
            return True, "drop success but still dropping"
        else:
            return False, "I don't know what happened but the drop failed"

    env.controller.do_action(agent_idx=0, action="turn_to", params={"target": container})
    env.controller.next_key_frame() # this action cannot fail (seems like)
    
    top = env.controller.manager.objects[container].top()
    top = top + np.array([0, 0.2, 0])
    env.controller.do_action(agent_idx=0, action="reach_for", params={"target": TDWUtils.array_to_vector3(top),
                                                                    "absolute": True,
                                                                    "offhand_follows": False,
                                                                    "arm": Arm.right,
                                                                    "from_held": True,
                                                                    "held_point": "top",
                                                                    "max_distance": 2.5,
                                                                    "arrived_at": 0.05,
                                                                    "plan": IkPlanType.vertical_horizontal})
    status = env.controller.next_key_frame()[0][0]
    reach_success = True
    if status != ActionStatus.success:
        env.controller.do_action(agent_idx=0, action="reset_arm",
                                 params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
        env.controller.next_key_frame()
        reach_success = False
        if env_type == "wind": return False, "failed to reach"
    # if status != ActionStatus.success:
    #     return False, "failed to reach"
    
    if env_type == "wind":
        env.controller.manager.settled.add(grasp_id)
    
    env.controller.do_action(agent_idx=0, action="drop", params={"arm": arm, "max_num_frames": 20})
    status = env.controller.next_key_frame()[0][0]
    env.controller.do_action(agent_idx=0, action="reset_arm", params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
    env.controller.next_key_frame()

    if status == ActionStatus.success:
            return True, f"drop success, reach for {'success' if reach_success else 'failed, may cause problems'}"
    elif status == ActionStatus.still_dropping:
        return True, f"drop success but still dropping, reach for {'success' if reach_success else 'failed, may cause problems'}"
    else:
        return False, "I don't not what happened but the drop failed"
    

def agent_inference(env, inference_time: float, fps: int = 30):
    """
    Advance the TDW sim for the duration of your recorded inference time,
    without executing any agent action. This keeps the world animating
    while your model "thinks."

    Args:
        env: your TDW environment wrapper
        inference_time: elapsed time (in seconds) that the agent took to infer
        fps: simulation frames per second (default: 30)

    Returns:
        (True, "success")
    """
    frame_interval = 1.0 / fps
    start_frame = env.controller.frame_count
    total_frames = int(inference_time * fps)
    ending_frame = start_frame + total_frames

    while env.controller.frame_count <= ending_frame:
        env.controller.next_key_frame()
        print("paused time")
        time.sleep(frame_interval)

    return True, "success"

def id_reverse_renumbering(env, reverse_id):
        for id in env.controller.manager.id_renumbering:
            if env.controller.manager.id_renumbering[id] == reverse_id:
                return id
        return -1

def update_object_status(obs, current_seen_objects_id):
    env_change_record = {}
    for o_id in current_seen_objects_id:
        obj_id = int(o_id)
        obj_mask = (obs["raw"]["seg_mask"] == obj_id)
        # import pdb; pdb.set_trace()
        if type(obj_mask) != np.ndarray:
            temp = obs["raw"]["log_temp"] * obj_mask.cpu().numpy()
            avg_temp = (temp.sum() / obj_mask.sum()).item()
        else:
            temp = obs["raw"]["log_temp"] * obj_mask
            avg_temp = temp.sum() / obj_mask.sum()
        if obj_id not in env_change_record:
            env_change_record[str(obj_id)] = [avg_temp]
        else:
            env_change_record[str(obj_id)].append(avg_temp)
    return env_change_record


def get_target_info( target_list):
        value_dict = json.load(open(os.path.join(PATH, "scenes/scene_configs/value.json")))
        object_attribute_dict = {}
        for target_category in target_list:
            object_attribute_dict[target_category] = {}
        for target_category in target_list:
            if target_category in value_dict:
                object_attribute_dict[target_category]['value'] = 5 if value_dict[target_category] == 1 else 1
            else:
                object_attribute_dict[target_category]['value'] = 1
        fireproof_dict = json.load(open(os.path.join(PATH, "scenes/scene_configs/fire.json")))
        for target_category in target_list:
            if target_category in fireproof_dict:
                object_attribute_dict[target_category]['fireproof'] = fireproof_dict[target_category]
            else:
                object_attribute_dict[target_category]['fireproof'] = 0
        return object_attribute_dict


def feedback_decision(feedback):
    """
    feedback_decision function takes the response from the OpenAI API and determines the next action based on the feedback.
    """
    default_result = True, "continue"
    result = default_result

    if "Explore" in feedback:
        result = False, ("explore",0)
    id = extract_numbers(feedback)
    if "PICK UP" in feedback and len(id) > 0:
        result = False, ("pick_up", int(id[0]))
    if "Inappropriate" in feedback and len(id) > 0:
        result = False, ("walk_to", int(id[0]))

    if result[1][0] == "pick_up":
        return result
    # weighted_choice_result  = weighted_choice(result, default_result)
    return result

def extract_numbers(s):
    # Find all digit sequences in the string
    numbers = re.findall(r'\d+', s)
    # Convert them to integers
    return [int(num) for num in numbers]

def weighted_choice(option1, option2, prob_option1=1):
    result = random.choices([1, 0], weights=[prob_option1, 1 - prob_option1])[0]
    if result == 1:
        return option1
    else:
        return option2


def llm_feedback(env, target_object_id, last_action_info, previous_seen,task="fire"):
    prompt_template_path = "/home/yangqingzheng/HAZARD/src/HAZARD/policy/llm_configs/feedback_prompt.csv"
    df = pd.read_csv(prompt_template_path)
    df.set_index('type', inplace=True)
    prompt = df.loc['fire', 'prompt']
    obs = env.controller._obs()
    sem_map = obs["sem_map"]

    # def get_object_location_description(self):
    object_location_description_list = []
    id_map = sem_map['explored'] * sem_map['id']
    explored_object_id_list = [int(idx) for idx in set(id_map.flatten()) if int(idx) != 0]
    explored_object_id_list = list(set(explored_object_id_list))
    explored_object_name_list = [env.controller.manager.segm.names[id_reverse_renumbering(env, idx)]
                                     for idx in explored_object_id_list]
    explored_object_category_list = [env.controller.manager.segm.categories[id_reverse_renumbering(env, idx)]
                                         for idx in explored_object_id_list]
    objects_list = [
            {'name': name, 'category': category, 'id': str(idx)} for idx, name, category in zip(explored_object_id_list,
                                                                          explored_object_name_list,
                                                                          explored_object_category_list) if idx in id_map
        ]
    object_ids = [int(obj['id']) for obj in objects_list]
    object_centers = []
    
    # def get_object_center(self):
    for idx in object_ids:
        object_points = (id_map == idx).nonzero()
        if type(object_points[0]) == np.ndarray:
            object_centers.append((object_points[0].astype(float).mean(),
                                    object_points[1].astype(float).mean()))
        else:
            object_centers.append((object_points[:, 0].float().mean().item(),
                                    object_points[:, 1].float().mean().item()))
        agent_map = obs["goal_map"]
        agent_points = (agent_map == -2).nonzero()
        if type(agent_points[0]) == np.ndarray:
            agent_pos = (agent_points[0].astype(float).mean(), agent_points[1].astype(float).mean())
        else:
            agent_pos = (agent_points[:, 0].float().mean().item(), agent_points[:, 1].float().mean().item())
        # TODO use relative location to agent
        for center in object_centers:
            distance = np.array((agent_pos[0] - center[0], agent_pos[1] - center[1]))
            distance = np.linalg.norm(distance)
            # description = f"distance from me: {round(distance, 2)}"
            description = f"{round(distance, 2)}"
            object_location_description_list.append(description)

    ps = ""
    current_seen_objects_id = [str(x) for x in list(set(obs["raw"]['seg_mask'].flatten()))]
    target_objects = env.controller.target
    target_info = get_target_info(target_objects)
    
    for obj, desc in zip(objects_list, object_location_description_list):
        if obj['category'] not in target_objects or obj['id'] not in current_seen_objects_id:
            continue
        ps += f"name: {obj['category']}, id: {obj['id']}, value: {target_info[obj['category']]['value']}, distance: {desc} m, "
        if task == "fire":
            env_change_record = update_object_status(obs, current_seen_objects_id)
            if obj['id'] not in env_change_record:
                ps += f"temperature: unknown\n"
            else:
                ps += f"temperature: {str(round(np.exp(env_change_record[obj['id']][-1]), 2))} Celsius\n"

    # previous action 
    # action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)

    prompt = prompt.replace('$CURRENT_ACTION$', '(walk to,' + str(target_object_id) + ")" + "\n")
    prompt = prompt.replace('$OBS$', ps + "\n")
    prompt = prompt.replace('$PRE$', last_action_info + "\n")
    prompt = prompt.replace('$PRE_TAR$', str(previous_seen) + "\n")
    print("prompt:", prompt)
    # resp = client.chat.completions.create(model='gpt-5',
    # messages=[
    #     {"role":"system", "content":"You are an expert roboticist., " },
    #     {"role":"user",   "content":prompt}
    # ],
    # temperature=0.7,
    # max_tokens=64)
    # gpt 5 
    resp = client.chat.completions.create(model='gpt-5',
                messages=[
                    {"role":"system", "content":"You are an expert roboticist., " },
                    {"role":"user",   "content":prompt}
                ],
                max_completion_tokens = 64,
                verbosity = "medium",
                reasoning_effort = "minimal",
                n = 1,
    )
    feedback = resp.choices[0].message.content
    print("feedback decision from llm prompt:", feedback)
    feedback_des, action = feedback_decision(feedback)
    print("action:", action)
    return feedback_des, action, ps