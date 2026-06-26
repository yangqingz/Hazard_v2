import shutil

import os
import json
import numpy as np
import time
from .envs.flood import FloodEnv
from .envs.fire import FireEnv
from .envs.wind import WindEnv
from .envs.flood.utils import ObjectState as FloodObjectState
from .envs.fire.fire_utils import ObjectState as FireObjectState
from .policy.env_actions import (agent_walk_to, agent_pickup, agent_drop, agent_explore, visualize_obs,
                                 agent_walk_to_single_step, low_level_action, agent_inference)
import logging
from src.HAZARD.policy.astar import get_astar_path, get_astar_weight

PATH = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(PATH) != "src":
    PATH = os.path.dirname(PATH)
PATH = os.path.dirname(PATH)


def get_target_description(env):
    return env.controller.target

def get_target_name(env):
    return env.controller.target_names

def get_target_ids(env):
    return env.controller.target_ids

def init_logs(output_dir, name = 'simple_example'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class Challenge:
    def __init__(self, env_name, data_dir, output_dir, logger, launch_build=True, port=1071, screen_size=512,
                 map_size_h=512, map_size_v=512, grid_size=0.1, debug=False, max_steps=1500, use_gt=False,
                 reverse_observation=False, record_only=False, record_with_agents=False, use_dino=False,
                 effect_on_agents=False, use_cached_assets=False, use_dstar=False):
        if env_name == "fire":
            env = FireEnv
            max_steps = 1500 if not record_only else 4500
        elif env_name == "flood":
            env = FloodEnv
            max_steps = 1500
        elif env_name == "wind":
            env = WindEnv
            max_steps = 3000
        else:
            assert False
        self.env_name = env_name
        self.effect_on_agents = effect_on_agents
        if debug:
            self.env = env(launch_build=True, screen_size=screen_size, port=port, use_local_resources=use_cached_assets,
                           map_size_h=map_size_h, map_size_v=map_size_v, grid_size=grid_size,
                           image_capture_path=os.path.join(output_dir, "images"), use_dino=use_dino,
                           log_path = os.path.join(output_dir, "log.txt"), reverse_observation=reverse_observation,
                           check_version=False, use_gt=use_gt, record_only=record_only)
        else:
            self.env = env(launch_build=True, screen_size=screen_size, port=port, use_local_resources=use_cached_assets,
                           map_size_h=map_size_h, map_size_v=map_size_v, grid_size=grid_size,
                           image_capture_path=os.path.join(output_dir, "images") if record_with_agents else None,
                           check_version=False, use_gt=use_gt, use_dino=use_dino,
                           log_path=os.path.join(output_dir, "log.txt"),
                           reverse_observation=reverse_observation, record_only=record_only)
        self.logger = logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.debug = debug
        self.output_parent_dir = output_dir
        self.output_dir = output_dir
        self.data_dir = data_dir
        #self.env.reset(data_dir=data_dir)
        self.logger.info("done")
        self.holding_object = []
        self.nearest_object = None
        self.have_finished_list = []
        self.high_value = 5
        self.low_value = 1
        self.max_steps = max_steps
        self.record_with_agents = record_with_agents
        self.use_dstar = use_dstar
        self.pickup_stats = {}  # Add this to track per-object stats

    def reset(self):
        self.holding_object = []
        self.nearest_object = None
        self.have_finished_list = []
        self.pickup_stats = {}  # Reset stats on episode reset

    def get_target_info(self, target_list):
        value_dict = json.load(open(os.path.join(PATH, "src/HAZARD/scenes/scene_configs/value.json")))
        object_attribute_dict = {}
        for target_category in target_list:
            object_attribute_dict[target_category] = {}
        for target_category in target_list:
            if target_category in value_dict:
                object_attribute_dict[target_category]['value'] = self.high_value if value_dict[target_category] == 1 else self.low_value
            else:
                object_attribute_dict[target_category]['value'] = self.low_value
        if self.env_name == 'fire':
            fireproof_dict = json.load(open(os.path.join(PATH, "src/HAZARD/scenes/scene_configs/fire.json")))
            for target_category in target_list:
                if target_category in fireproof_dict:
                    object_attribute_dict[target_category]['fireproof'] = fireproof_dict[target_category]
                else:
                    object_attribute_dict[target_category]['fireproof'] = 0
        elif self.env_name == 'flood':
            waterproof_dict = json.load(open(os.path.join(PATH, "src/HAZARD/scenes/scene_configs/fluid.json")))
            for target_category in target_list:
                if target_category in waterproof_dict:
                    object_attribute_dict[target_category]['waterproof'] = waterproof_dict[target_category]
                else:
                    object_attribute_dict[target_category]['waterproof'] = 0
        return object_attribute_dict

    def id_renumbering(self, id):
        return self.env.controller.manager.id_renumbering[id]

    def id_reverse_renumbering(self, reverse_id):
        for id in self.env.controller.manager.id_renumbering:
            if self.env.controller.manager.id_renumbering[id] == reverse_id:
                return id
        return -1

    def process_input(self, state, action_result, action_info, distance=0.0):
        processed_input = {}
        # Get visible objects from semantic map
        explored_sem_id_map = state['sem_map']['explored']*state['sem_map']['id']
        explored_object_id_list = [int(idx) for idx in set(explored_sem_id_map.flatten()) if int(idx) != 0]
        explored_object_id_list = list(set(explored_object_id_list))
        explored_object_name_list = [self.env.controller.manager.segm.names[self.id_reverse_renumbering(idx)]
                                     for idx in explored_object_id_list]
        explored_object_category_list = [self.env.controller.manager.segm.categories[self.id_reverse_renumbering(idx)]
                                         for idx in explored_object_id_list]
        id_map = set((state['sem_map']['explored'] * state['sem_map']['id']).flatten())
        processed_input['explored_object_name_list'] = [
            {'name': name, 'category': category, 'id': str(idx)} for idx, name, category in zip(explored_object_id_list,
                                                                          explored_object_name_list,
                                                                          explored_object_category_list) if idx in id_map
        ]
        processed_input['holding_objects'] = self.holding_object
        # Add pickup statistics to processed input
        processed_input['pickup_stats'] = self.pickup_stats
        processed_input['previous_distance_walked'] = distance  # Add total distance
        if self.nearest_object != None:
            try:
                processed_input['nearest_object'] = [
                    {'name': self.env.controller.manager.segm.names[self.id_reverse_renumbering(self.nearest_object)],
                     'category': self.env.controller.manager.segm.categories[
                         self.id_reverse_renumbering(self.nearest_object)],
                     'id': str(self.nearest_object)}]
            except:
                processed_input['nearest_object'] = []
        else:
            processed_input['nearest_object'] = []
        processed_input['step'] = self.env.controller.frame_count
        processed_input['action_result'] = action_result
        processed_input['action_info'] = action_info
        # visualize_obs(self.env, None, str(self.step_num), "output")
        return processed_input

    def hold_object(self):
        self.holding_object.append({'name': self.env.controller.manager.segm.names[self.id_reverse_renumbering(self.nearest_object)],
                                    'category': self.env.controller.manager.segm.categories[self.id_reverse_renumbering(self.nearest_object)],
                                    'id': str(self.nearest_object)})
        self.nearest_object = None

    def drop_object(self):
        obj_id = int(self.holding_object[0]['id'])
        reversed_id = self.id_reverse_renumbering(obj_id)
        # print(reversed_id)
        if reversed_id in self.target_status:
            # print('ok')
            # Can not pick up again, because this target is finished
            self.target_status[reversed_id] = self.env.controller.frame_count
            self.have_finished_list.append(obj_id)
        else:
            # print('not ok')
            self.nearest_object = obj_id
        self.holding_object = []

    def get_score(self):
        total_score = 0
        max_score = 0
        value_dict = json.load(open("src/HAZARD/data/meta_data/value.json"))
        self.final_states = dict()
        if self.env_name in ["fire", "flood"]:
            waterproof_dict = json.load(open("src/HAZARD/scenes/scene_configs/fluid.json"))
            print("finally:", self.target_status)
            for target in self.target_status:
                name = self.env.controller.target_id2name[target]
                if name in value_dict:
                    if value_dict[name] == 1:
                        value = self.high_value
                    else:
                        value = self.low_value
                else:
                    value = self.low_value
                if self.target_status[target]:
                    print("picking up target:", name)
                    if self.env_name == "fire":
                        if self.env.controller.manager.objects[target].state == FireObjectState.NORMAL:
                            total_score += value
                            print("normal, so total score:", total_score)
                        else:
                            total_score += value * 0.5
                            print("not normal, so total score:", total_score)
                        self.final_states[target] = self.env.controller.manager.objects[target].state.value
                    elif self.env_name == "flood":
                        if name in waterproof_dict:
                            waterproof = waterproof_dict[name]
                        else:
                            waterproof = 0
                        if waterproof or \
                                self.env.controller.manager.objects[target].state == FloodObjectState.NORMAL or \
                                self.env.controller.manager.objects[target].state == FloodObjectState.FLOATING:
                            total_score += value
                        else:
                            total_score += value * 0.5
                        self.final_states[target] = self.env.controller.manager.objects[target].state.value
                    else:
                        total_score += value
                        self.final_states[target] = 0
                max_score += value
                print("target:", name, 'with value:', value, 'total score:', total_score, 'max score:', max_score)
        else:
            total_score = 0
            max_score = 1
        return total_score, max_score

    def submit(self, agent, logger, eval_episodes, inference=False):
        total_finish = 0.0
        total_score = 0.0
        total_max_score = 0.0
        total_steps = 0
        num_eval_episodes = eval_episodes

        start = time.time()
        print("OUTPUT_DIR", self.output_dir)
        for i in range(num_eval_episodes):
            start_time = time.time()
            if not os.path.exists(os.path.join(self.output_dir, str(i))):
                os.makedirs(os.path.join(self.output_dir, str(i)))
            self.logger.info('Episode: {}/{}'.format(i + 1, num_eval_episodes))
            self.logger.info(f"Resetting Environment ... data is {self.data_dir}")
            self.reset()
            self.env.reset(data_dir=self.data_dir)
        #    camera = ThirdPersonCamera(avatar_id="a", position={"x": -0.9, "y": 2.0, "z": 2.3}, look_at=1194112)
        #    self.env.controller.add_ons.append(camera)
            target_description = get_target_description(self.env)
            target_ids = get_target_ids(self.env)
            self.target_status = {target_id: False for target_id in target_ids}
            print('init:', self.target_status)
            target_info = self.get_target_info(get_target_description(self.env))
            if target_description is not None:
                if agent.agent_type in ['greedy', 'llm', 'llmv2', 'mcts', 'mctsv2', 'human', 'record', 'rl', 'rule',
                                        'random', 'custom', 'llmv4','llm_predictor']:
                    agent.reset(goal_objects=target_description,
                                objects_info=target_info)
                elif agent.agent_type == 'oracle':
                    agent.reset(goal_objects=target_description,
                                objects_info=target_info,
                                controller=self.env.controller,
                                step_limit=self.max_steps)
                else:
                    raise Exception(f"{agent.agent_type} not available")
            else:
                assert False
                agent.reset(output_dir=os.path.join(self.output_dir, str(i)))
            # for debug

            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")
            local_finish = self.env.done
            done = False
            self.step_num = 0
            local_reward = 0.0
            action_result = False
            action_info = ""
            total_steps_to_pickup = 0
            Feedback = True
            current_action = None 
            last_action_info = ""
            previous_seen = ""
            self.env.controller.communicate([])
            if "demo" in self.output_dir:
                for i in range(1500):
                    self.env.controller.communicate([])
                return

            action_logger = open(os.path.join(self.output_dir, "actions.txt"), "w")

            if agent.agent_type == "record":
                while self.env.controller.frame_count < self.max_steps:
                    self.env.controller.communicate([])
                done = True

            while not done:
                # self.env.controller._done()
                if self.env_name in ["fire", "flood"]:
                   print("Target status:")
                   for target in self.target_status:
                        print(target, self.target_status[target], self.env.controller.manager.objects[target].state, self.env.controller.manager.objects[target].position)
                state = self.env.controller._obs()
                # Suppose agent can not see the finished object
                for finished_id in self.have_finished_list:
                    state['sem_map']['id'][state['sem_map']['id'] == finished_id] = 0
                    state['raw']['seg_mask'][state['raw']['seg_mask'] == finished_id] = 0
                self.step_num += 1
                processed_input = self.process_input(state, action_result, action_info)
                processed_input['save_dir'] = str(os.path.join(self.output_dir, str(i)))
                
                print("process pickup stats:", processed_input['pickup_stats'])
                # print(path_info)
                if agent.agent_type == "llm" or agent.agent_type == "llmv2" or agent.agent_type == "llmv4":
                    import json
                    with open(os.path.join(self.output_dir, str(i), f"input{self.env.controller.frame_count}.json"), "w") as f:
                        json.dump(processed_input, f, indent=4)
                    visualize_obs(self.env, state, suffix=str(self.env.controller.frame_count), save_dir=os.path.join(self.output_dir, str(i)))
                if agent.agent_type == "mcts" or agent.agent_type == "mctsv2":
                    visualize_obs(self.env, state, suffix=str(self.env.controller.frame_count),
                                  save_dir=os.path.join(self.output_dir, str(i)))
                

                ## main function to get action from agent
                start_time = time.time()
                potential_action = agent.choose_target(state, processed_input)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("potential action ISSSSSSSSSS: ", potential_action)

                # This part is for updating the frame while the agent is doing inference
                if inference:
                    agent_inference(self.env, elapsed_time)
                    print("inference time", elapsed_time)

                # Feedback is used to indicate whether the agent needs to wait for feedback from the environment
                # If Feedback is True, the agent will wait for the next step to get the feedback
                # If Feedback is False, the agent will execute the action immediately
                print("index 0", potential_action[0])
                if Feedback == False or potential_action[0] == "pick_up" or potential_action[0] == "drop" or current_action == None:
                    current_action = potential_action
                    Feedback = False
                else:
                    print("NOOO PONTENTIAL ACTION USED:< BASED ON FEEDBACK:", current_action)
                    Feedback = True
                
                if isinstance(current_action, int) and agent.agent_type in ["rl", "random"]:
                    current_action = self.env.get_challenge_action(current_action)
                print(current_action)
                if current_action[0] == "pick_up":
                    try: 
                        obj_name = self.env.controller.id2name[self.id_reverse_renumbering(self.nearest_object)]
                        print(f"step {self.env.controller.comm_counter} action {current_action} {obj_name} "
                            f"{str(self.id_reverse_renumbering(self.nearest_object))}", file=action_logger)
                        Feedback = False
                    except Exception as e:
                        print("should pick up", self.nearest_object, " with a id", self.id_reverse_renumbering(self.nearest_object),file=action_logger)

                else:
                    print(f"step {self.env.controller.comm_counter} action {current_action} elapsed time {elapsed_time:2f}", file=action_logger)

                if agent.agent_type == 'oracle':
                    while self.env.controller.frame_count < self.max_steps:
                        print(self.env.controller.frame_count, self.max_steps)
                        agent.save_info()
                        self.env.controller.communicate([])
                    oracle_plan = agent.search_plan()
                    import json
                    json.dump(oracle_plan, open(os.path.join(self.output_dir, f"action-{str(i)}.json"), "w"))
                elif current_action[0] == "walk_to":
                    # Initialize stats for new target
                    if int(current_action[1]) not in self.pickup_stats:
                        self.pickup_stats[int(current_action[1])] = {
                            "steps_to_pickup": 0,
                            "pickup_success": False,
                            "astar_len": self.get_object_astar_path(int(current_action[1]))
                        }
                        
                    # Update stats for this target
                    self.pickup_stats[int(current_action[1])]["steps_to_pickup"] += 1
                
                    if self.record_with_agents:
                        action_result, action_info, distance, previous_seen = agent_walk_to(self.env, target=self.id_reverse_renumbering(int(current_action[1])),
                                                               max_steps=100, reset_arms=False, arrived_at=1, feedback=Feedback,
                                                               task=self.env_name, record_mode=True,target_object_id=int(current_action[1]),last_action_info=last_action_info,
                                                               effect_on_agents=self.effect_on_agents, use_dstar=self.use_dstar, previous_seen=previous_seen)
                    elif self.env_name in ["fire", "flood"]:
                        action_result, action_info, distance, previous_seen = agent_walk_to(self.env, target=self.id_reverse_renumbering(int(current_action[1])), feedback=Feedback,
                                                           max_steps=100, reset_arms=False, arrived_at=1, task=self.env_name,target_object_id=int(current_action[1]),last_action_info=last_action_info,
                                                               effect_on_agents=self.effect_on_agents, use_dstar=self.use_dstar, previous_seen=previous_seen)
                    else:
                        action_result, action_info, distance, previous_seen = agent_walk_to(self.env, target=self.id_reverse_renumbering(int(current_action[1])),
                                                           max_steps=100, reset_arms=False, arrived_at=1, task=self.env_name,target_object_id=int(current_action[1]),last_action_info=last_action_info,
                                                               effect_on_agents=self.effect_on_agents, use_dstar=self.use_dstar, previous_seen=previous_seen, feedback=Feedback)
                    total_steps_to_pickup += 1

                    if action_result == True:
                        self.nearest_object = int(current_action[1])
                    else:
                        self.nearest_object = None
                elif current_action[0].startswith("low_level"):
                    action = current_action[0].split(".")[-1]
                    kwargs = current_action[1]
                    action_result, action_info = low_level_action(self.env, effect_on_agents=self.effect_on_agents,
                                                                  task=self.env_name, action=action, **kwargs)
                    if action_result and action == "move_by":
                        self.nearest_object = self.env.controller.find_nearest_object()
                    else:
                        self.nearest_object = None
                elif current_action[0] == "walk_to_single":
                    if self.env_name in ["fire", "flood"]:
                        action_result, action_info= agent_walk_to_single_step(self.env, target=self.id_reverse_renumbering(int(current_action[1])),
                                                               reset_arms=False, arrived_at=0.1, task=self.env_name,
                                                                               effect_on_agents=self.effect_on_agents,
                                                                               record_mode=self.record_with_agents)
                    else:
                        action_result, action_info = agent_walk_to_single_step(self.env, target=self.id_reverse_renumbering(int(current_action[1])),
                                                               reset_arms=False, arrived_at=0.1, task=self.env_name,
                                                                               effect_on_agents=self.effect_on_agents,
                                                                               record_mode=self.record_with_agents)
                    if action_result and action_info == "success":
                        self.nearest_object = int(current_action[1])
                    else:
                        self.nearest_object = None
                elif current_action[0] == "pick_up":
                    if current_action[1] is None:
                        print('WARNING: not specifying object id, will pick up the nearest object')
                        if self.nearest_object == None:
                            action_result, action_info = False, "You need to walk to an object first."
                        else:
                            action_result, action_info = agent_pickup(self.env, self.id_reverse_renumbering(self.nearest_object), env_type=self.env_name)
                    else:
                        self.nearest_object = int(current_action[1])
                        if int(current_action[1]) in self.pickup_stats and action_result:
                            self.pickup_stats[int(current_action[1])]["pickup_success"] = True
                        action_result, action_info = agent_pickup(self.env, self.id_reverse_renumbering(self.nearest_object), env_type=self.env_name)
                    if action_result:
                        self.hold_object()
                elif current_action[0] == "drop":
                    if current_action[1] is None and self.env_name == "wind" and self.id_reverse_renumbering(self.nearest_object) in self.env.controller.containers:
                        current_action = ("drop", self.nearest_object)
                    if current_action[1] is None:
                        print('WARNING: not specifying object id, will drop to the ground')
                        action_result, action_info = agent_drop(self.env, env_type=self.env_name)
                    else:
                        action_result, action_info = agent_drop(self.env, self.id_reverse_renumbering(int(current_action[1])), env_type=self.env_name)
                    if action_result:
                        self.drop_object()
                elif current_action[0] == "explore":
                    action_result, action_info = agent_explore(self.env)
                elif current_action[0] == "stop":
                    action_result, action_info = True, "stopped"
                    done = True
                elif current_action[0] == "record":
                    while self.env.controller.frame_count < self.max_steps:
                        self.env.controller.communicate([])
                else:
                    assert False, f"action {current_action} not available"
                local_finish = "success" if action_result else f"fail, because {action_info}"
                
                if action_result == 'feedback':
                    last_action_info = "The previous action is " + str(current_action)
                    current_action = action_info
                    Feedback = True
                    print("change the action to", current_action)
                else:
                    Feedback = False
                    last_action_info = str(current_action) + " " + str(action_result) +" because " + str(action_info)
                print("Feedback is", Feedback)
                self.logger.info(
                    f"Executing step {self.step_num} for episode: {i}, action: {current_action}, finish: {local_finish}, elapsed_time: {elapsed_time:.2f}")

                if self.env.controller.frame_count >= self.max_steps:
                    done = True
                print(self.env.controller.frame_count, self.max_steps)
                have_target_left = False
                for target in self.target_status:
                    if not self.target_status[target]:
                        have_target_left = True
                if not have_target_left:
                    done = True
                if done:
                    break
            action_logger.close()
            score, max_score = self.get_score()
            total_score += score
            total_max_score += max_score
            step = self.env.controller.frame_count
            total_steps += step
            if os.path.isfile(os.path.join(self.output_parent_dir, "log.txt")):
                print("    DEBUG Moving")
                if os.path.exists(os.path.join(self.output_parent_dir, "log.txt")):
                    shutil.move(os.path.join(self.output_parent_dir, "log.txt"), os.path.join(self.output_dir, f"log.txt"))
                if os.path.exists(os.path.join(self.output_parent_dir, "images")):
                    shutil.move(os.path.join(self.output_parent_dir, "images"), os.path.join(self.output_dir, f"images"))
            # with open(os.path.join(self.output_dir, str(i), 'result_episode.json'), 'w') as f:
            #     json.dump(result, f)
        avg_score = total_score / num_eval_episodes
        avg_steps = total_steps / num_eval_episodes
        avg_max_score = total_max_score / num_eval_episodes
        results = {
            "avg_score": avg_score,
            "avg_steps": avg_steps,
            "avg_max_score": avg_max_score,
            "total": num_eval_episodes,
            "target_status": self.target_status,
            "final_states": self.final_states,
        }
        import json
        with open(os.path.join(self.output_dir, 'eval_result.json'), 'w') as f:
            json.dump(results, f)
        self.logger.info(f'eval done, avg score {avg_score}, max score {avg_max_score}, avg steps {avg_steps}')
        # self.logger.info('eval done, avg reward {}, avg_finish {}'.format(avg_reward, avg_finish))
        self.logger.info('time: {}'.format(time.time() - start))
        return avg_score / avg_max_score, avg_steps

    def close(self):
        self.env.close()

    
    def get_object_astar_path(self, object_id, arrived_at: float = 0.5):
        """
        Calculate A* path and distance to a single object
        Args:
            object_id: ID of the target object
        Returns:
            Dictionary containing serializable path information
        """
        # Get observation and semantic map
        obs = self.env.controller._obs()
        sem_map = obs["sem_map"]

        # Get explored area mask and ID map
        explored = sem_map["explored"]
        id_map = explored * sem_map["id"]

        # Get agent position and convert to native Python types
        agent_real = np.asarray(self.env.controller.agents[0].dynamic.transform.position, dtype=float)
        agent_grid = tuple(map(int, self.env.controller.real_to_grid(agent_real)))

        # Find object center in grid coordinates
        coords = np.argwhere(id_map == object_id)
        if coords.shape[0] == 0:
            return {
                "agent": {
                    "pos_world": [float(x) for x in agent_real],
                    "pos_grid": agent_grid,
                },
                "object": None,
                "error": "Object not found in semantic map"
            }

        # Find closest point and convert to native Python types
        dists = np.linalg.norm(coords - agent_grid, axis=1)
        target_grid = tuple(map(int, coords[dists.argmin()]))

        # Try to compute A* path
        try:
            weight = get_astar_weight(sem_map=sem_map, origin=agent_grid, destination=target_grid)
            path = get_astar_path(weight=weight, origin=agent_grid, destination=target_grid)
            
            astar_len = None
            if path is not None and len(path) > 1:
                astar_len = 0.0
                pruned_path = []
                for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                    # Convert coordinates to native Python types
                    step = float(np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1])))
                    astar_len += step
                    pruned_path.append((int(x1), int(y1)))
                    
                    if np.linalg.norm(np.array([x2, y2]) - np.array(target_grid)) <= arrived_at:
                        pruned_path.append((int(x2), int(y2)))
                        break
            
            path = pruned_path
            astar_len = float(astar_len)  # Convert to native Python float
            
        except Exception as e:
            astar_len = None
            path = None

        return astar_len