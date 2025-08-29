import pdb
import tiktoken
from openai import OpenAI

client = OpenAI()
import json
import os
import pandas as pd
import numpy as np
import random
import re
from openai import OpenAIError
import backoff
import time
import torch
from src.HAZARD.policy.astar import get_astar_path, get_astar_weight

class SamplingParameters:
    def __init__(self, debug=False, max_tokens=64, t=0.7, top_p=1.0, n=1, logprobs=1, echo=False):
        self.debug = debug
        self.max_tokens = max_tokens
        self.t = t
        self.top_p = top_p
        self.n = n
        self.logprobs = logprobs
        self.echo = echo


class LLMv4:
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
                 lm_id,
                 prompt_template_path,
                 cot,
                 sampling_parameters,
                 task,
                 api_key,
                 model_and_tokenizer_path="",
                 total_max_tokens=4096,
                 max_plan_idx=3  # Add this parameter
                 ):
        self.env_change_record = None
        if type(api_key) == list:
            self.apikey_list = api_key
        else:
            self.apikey_list = [api_key]
        self.model_and_tokenizer_path = model_and_tokenizer_path
        self.apikey_idx = 0
        self.agent_type = "llmv4"
        self.task = task
        assert task in ['fire', 'flood', 'wind']
        self.debug = sampling_parameters.debug
        self.prompt_template_path = prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        df.set_index('type', inplace=True)
        self.prompt_template = df.loc[self.task, 'prompt']
        self.cot = cot
        self.source = source
        self.lm_id = lm_id
        self.chat = any(tok in lm_id for tok in ['gpt-3.5-turbo', 'gpt-4', 'o1-preview', 'gpt-4.1-nano'])
        self.total_cost = 0
        self.total_max_tokens = total_max_tokens - sampling_parameters.max_tokens
        self.picked_up_objects = []
        if self.source == 'openai':
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.lm_id)
            except Exception:
                # o1-preview (and most chat models) use cl100k_base
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            if self.lm_id == 'o1-preview':
                self.sampling_params = {
                    "temperature": 1,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            elif self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif self.source == 'huggingface':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            assert model_and_tokenizer_path != ""
            # model_name = "meta-llama/Llama-2-7b-chat-hf"
            self.model = AutoModelForCausalLM.from_pretrained(model_and_tokenizer_path, device_map="auto",
                                                              load_in_4bit=False,max_memory={0: "18GB", "cpu": "6GB"},
    torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizer_path, 
                                                           use_fast=True)
            self.sampling_params = {
                "max_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "n": sampling_parameters.n,
                "logprobs": sampling_parameters.logprobs,
                "echo": sampling_parameters.echo,
            }
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id):
            @backoff.on_exception(backoff.expo, OpenAIError)
            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    base_prompt = prompt[0]["content"]
                    tokens = self.tokenizer.encode(base_prompt)
                    while len(tokens) >= self.total_max_tokens:
                        base_prompt = self.cut_prompt(base_prompt)
                        tokens = self.tokenizer.encode(base_prompt)
                    prompt[0]["content"] = base_prompt
                    try:
                        if self.chat:
                            response = client.chat.completions.create(model=lm_id, messages=prompt, **sampling_params)
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"src/HAZARD/policy/llm_configs/chat_raw.json", 'a') as f:
                                    f.write(response.model_dump_json(indent=4)) 
                                    f.write('\n')
                            generated_samples = [response.choices[i].message.content for i in
                                                 range(sampling_params['n'])]
                            if 'gpt-4' in self.lm_id or 'o4-mini' in self.lm_id:
                                usage = response.usage.prompt_tokens * 0.03 / 1000 + response.usage.completion_tokens * 0.06 / 1000
                            elif 'gpt-3.5' in self.lm_id:
                                usage = response.usage.total_tokens * 0.002 / 1000
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 				  range(sampling_params['n'])]
                        elif "text-" in lm_id:
                            response = client.completions.create(model=lm_id, prompt=prompt, **sampling_params)
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"output/raw.json", 'a') as f:
                                    print(response.model_dump_json(indent=4))
                                    f.write('\n')
                            generated_samples = [response.choices[i].text for i in range(sampling_params['n'])]
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 			  range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except OpenAIError as e:
                        print(e)
                        print("Switch key and retry...")
                        self.sleep()
                        raise e
                elif source == 'huggingface':
                    if self.chat:
                        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                        model_inputs = self.tokenizer(prompt, return_tensors="pt")
                        while model_inputs.input_ids.shape[1] >= self.total_max_tokens:
                            prompt = self.cut_prompt(prompt)
                            model_inputs = self.tokenizer(prompt, return_tensors="pt")
                        model_inputs = model_inputs.to("cuda:0")
                        output = self.model.generate(**model_inputs, max_length=model_inputs.input_ids.shape[1]
                                                     + self.sampling_params["max_tokens"],
                                                     temperature=self.sampling_params['temperature'],
                                                     top_p=self.sampling_params['top_p'])
                        generated_samples = [self.tokenizer.decode(output[0][model_inputs.input_ids.shape[1]:],
                                                                   skip_special_tokens=True)]
                    else:
                        raise ValueError("Can not use non-chat huggingface models")
                else:
                    raise ValueError("invalid source")
                # generated_samples = [sample.strip().lower() for sample in generated_samples]
                return generated_samples, usage

            return _generate

        self.generator = lm_engine(self.source, self.lm_id)

        self.object_list = None
        self.holding_objects = None
        self.action_history = []
        self.action_history_result = []
        self.action_info_history = []
        self.agent_pos = None
        self.nearest_object = None
        self.objects_info = None
        self.target_objects = None
        self.current_seen_objects_id = None
        self.current_state = None
        self.current_step = None
        self.save_dir = None

        # Add new attributes for rescue planning
        self.max_plan_idx = max_plan_idx  # Store max_plan_idx
        self.rescue_plan = []
        self.current_plan_index = 0
        self.plan_complete = False

    def sleep(self, sleep_time=0.5):
        self.apikey_idx += 1
        if self.apikey_idx >= len(self.apikey_list):
            self.apikey_idx = 0
        time.sleep(sleep_time)

    def cut_prompt_with_given_positions(self, position1, position2, prompt):
        assert position1 in prompt
        prompt = prompt.split(position1)
        if len(prompt) < 2:
            return False, prompt
        head, tail = prompt
        head = head + position1
        assert position2 in tail
        tail = tail.split(position2)
        if len(tail) < 2:
            return False, prompt
        mid, tail = tail
        tail = tail + position2
        mid = mid.split("\n")
        if len(mid) > 0:
            mid = mid[:-1]
            mid = "\n".join(mid)
            prompt = head + mid + tail
            return True, prompt
        return False, prompt

    def cut_prompt(self, prompt):
        for pos1, pos2 in [("Objects states history:\n", "\nAvailable actions:"),
                           ("Target objects:\n", "\nCurrent State:"),
                           ("Previous actions:\n", "\nObjects states history:")]:
            success, prompt = self.cut_prompt_with_given_positions(position1=pos1, position2=pos2, prompt=prompt)
            if success:
                return prompt
        prompt = "\n".join(prompt.split("\n")[:-1])
        return prompt

    def update_history(self, action):
        self.action_history.append(action)

    def update_history_action_result(self, result, info):
        if len(self.action_history) == 0:
            return
        self.action_history_result.append(result)
        self.action_info_history.append(info)

    def reset(self, goal_objects, objects_info):
        self.target_objects = goal_objects
        self.objects_info = objects_info
        self.env_change_record = {}
        self.action_history = []
        self.action_history_result = []
        self.picked_up_objects = []
        self.rescue_plan = []
        self.current_plan_index = 0
        

    def objects_list2text(self):
        if self.debug:
            print(self.objects_info)
        s = '\n'.join([
            f"name: {category}, value: {str(self.objects_info[category]['value'])}, attribute: {('waterproof' if self.objects_info[category]['waterproof'] == 1 else 'non-waterproof') if self.task == 'flood' else 'None'}"
            for category in self.objects_info])

        return s

    def parse_answer(self, available_actions, text):
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action in text:
                return action

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(
                    ' ') or f"Option {option}" in text or f"({option})" in text or f"action {option}" in text or (
                    len(text) <= 2 and option in text):
                return action
        print(f"WARNING! Fuzzy match! Text: {text}")
        for i in range(len(available_actions)):
            action = available_actions[i]
            act = "None"
            name = "None"
            id = "None"
            if action.startswith('walk_to'):
                act = 'go to'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('pick_up'):
                act = 'pick'
            elif action.startswith('drop'):
                act = 'put'
            elif action.startswith('explore'):
                act = 'turn'
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action
        if len(text) == 1:
            i = ord(text) - ord('A')
            if i in range(len(available_actions)):
                return available_actions[i]
        print("WARNING! No available action parsed!!! Random choose one")
        return random.choice(available_actions)

    def progress2text(self):
        # s = f"I've taken {current_step}/3000 steps. "
        ##todo: add temp/height/... as current object state

        object_location_description_list = self.get_object_location_description()
        object_center_list = self.get_object_center()
        if self.task == 'wind':
            ps = "Shopping carts already found:\n"
            for obj, desc in zip(self.object_list, object_location_description_list):
                if obj['category'] != 'shopping cart':
                    continue
                ps += f"name: {obj['category']}, id: {obj['id']}, distance: {desc} m\n"
        else:
            ps = ""
        ps += "Target objects currently seen:\n"
        for obj, desc, center in zip(self.object_list, object_location_description_list, object_center_list):
            # print(type(obj['id']), type(self.current_seen_objects_id[0]))
            if obj['category'] not in self.target_objects or obj['id'] not in self.current_seen_objects_id:
                continue
            ps += f"name: {obj['category']}, id: {obj['id']}, value: {self.objects_info[obj['category']]['value']}, distance to agent: {desc} m, position at: ({center[0]} , {center[1]}) "
            if self.task == 'fire':
                if obj['id'] not in self.env_change_record:
                    ps += f"temperature: unknown\n"
                else:
                    ps += f"temperature: {str(round(np.exp(self.env_change_record[obj['id']][-1]), 2))} Celsius\n"
            elif self.task == 'flood':
                if obj['id'] not in self.env_change_record:
                    ps += f"water level: unknown\n"
                else:
                    ps += f"water level: {str(round(self.env_change_record[obj['id']][-1], 2))} m\n"
            else:
                ps += f"status: {'Unknown'}\n"
        ps += 'Target objects previously seen:\n'
        for obj, desc in zip(self.object_list, object_location_description_list):
            if obj['category'] not in self.target_objects or obj['id'] in self.current_seen_objects_id:
                continue
            ps += f"name: {obj['category']}, id: {obj['id']}, value: {self.objects_info[obj['category']]['value']}, distance: {desc} m, "
            if self.task == 'fire':
                if obj['id'] not in self.env_change_record:
                    ps += f"temperature: unknown\n"
                else:
                    ps += f"temperature: {str(round(np.exp(self.env_change_record[obj['id']][-1]), 2))} Celsius\n"
            elif self.task == 'flood':
                if obj['id'] not in self.env_change_record:
                    ps += f"water level: unknown\n"
                else:
                    ps += f"water level: {str(round(self.env_change_record[obj['id']][-1], 2))} m\n"
            else:
                ps += f"status: {'Unknown'}\n"
        return ps

    def get_object_location_description(self):
        object_location_description_list = []
        id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
        object_ids = [int(obj['id']) for obj in self.object_list]
        # list of object IDs

        # agent position in real + grid space
        agent_grid = self.agent_pos
        if not isinstance(agent_grid, (list, tuple, np.ndarray)) or len(agent_grid) < 2:
            print("[A*] ERROR: self.agent_pos malformed:", agent_grid)
            return ["planner error"]

        # ensure ints
        agent_grid = (int(round(agent_grid[0])), int(round(agent_grid[1])))

        for idx in object_ids:
            # object pixels in semantic map
            object_points = (id_map == idx).nonzero()

            # compute object center in grid coordinates
            if type(object_points[0]) == np.ndarray:  # numpy
                center = (object_points[0].astype(float).mean(),
                        object_points[1].astype(float).mean())
            else:  # torch tensor
                center = (object_points[:, 0].float().mean().item(),
                        object_points[:, 1].float().mean().item())

            target_grid = (int(round(center[0])), int(round(center[1])))

            # try to compute A* path
            try:
                weight = get_astar_weight(sem_map=self.current_state['sem_map'],
                                        origin=agent_grid,
                                        destination=target_grid)
                path = get_astar_path(weight=weight,
                                    origin=agent_grid,
                                    destination=target_grid)
                if path is not None and len(path) > 1:
                    # compute path length (grid units, with diagonals = sqrt(2))
                    dist = 0.0
                    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                        dx, dy = abs(x2 - x1), abs(y2 - y1)
                        dist += np.sqrt(2) if (dx == 1 and dy == 1) else 1.0

                    description = f"{round(dist, 2)}"
                else:
                    description = "unreachable"
            except Exception:
                description = "planner error"

            object_location_description_list.append(description)

        return object_location_description_list

    def update_object_status(self):
        for o_id in self.current_seen_objects_id:
            obj_id = int(o_id)
            if self.task in ["fire", "flood"]:
                obj_mask = (self.current_state["raw"]["seg_mask"] == obj_id)
                # import pdb; pdb.set_trace()
                if type(obj_mask) != np.ndarray:
                    temp = self.current_state["raw"]["log_temp"] * obj_mask.cpu().numpy()
                    avg_temp = (temp.sum() / obj_mask.sum()).item()
                else:
                    temp = self.current_state["raw"]["log_temp"] * obj_mask
                    avg_temp = temp.sum() / obj_mask.sum()
                if obj_id not in self.env_change_record:
                    self.env_change_record[str(obj_id)] = [avg_temp]
                else:
                    self.env_change_record[str(obj_id)].append(avg_temp)
            # if self.task == "fire":
            #     avg_temp = np.exp(avg_temp)
            #     text = f"object temperature is {str(round(avg_temp, 2))} Celsius"
            # else:
            #     text = f"water level at this object is {str(round(avg_temp, 2))} m"
        # else:
        # 	id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
        # 	object_points = (id_map == obj_id).nonzero()
        # 	if type(object_points[0]) == np.ndarray:
        # 		object_center = (object_points[0].astype(float).mean(), object_points[1].astype(float).mean())
        # 	else:
        # 		object_center = (object_points[:, 0].float().mean().item(), object_points[:, 1].float().mean().item())
        # 	if obj_id not in self.env_change_record:
        # 		self.env_change_record[obj_id] = [object_center]
        # 	else:
        # 		self.env_change_record[obj_id].append(object_center)

    def get_available_plans(self):
        available_plans = []
        actions = []
        if len(self.holding_objects) == 0:
            for obj in self.object_list:
                if obj['category'] not in self.target_objects or (
                        len(self.holding_objects) > 0 and obj['id'] == self.holding_objects[0]['id']):
                    continue
                action = ("walk_to", obj['id'])
                available_plans.append(f"go pick up object <{obj['category']}> ({obj['id']})")
                actions.append(action)
        else:
            if self.task == "wind":
                for obj in self.object_list:
                    if obj['category'] != 'shopping cart':
                        continue
                    action = ("walk_to", obj['id'])
                    available_plans.append(f"go put object into <{obj['category']}> ({obj['id']})")
                    actions.append(action)
            else:
                action = ("drop", None)
                available_plans.append(f"put the holding object in my bag")
                actions.append(action)

        if len(self.action_history) == 0 or self.action_history[-1] != 'look around':
            action = ("explore", None)
            available_plans.append("look around")
            actions.append(action)

        if len(actions) == 0:
            for obj in self.object_list:
                action = ("walk_to", obj['id'])
                available_plans.append(f"go to object <{obj['category']}> ({obj['id']})")
                actions.append(action)
        if len(actions) > 10:
            actions = actions[:10]
            available_plans = available_plans[:10]
        if len(actions) == 0:
            action = ("explore", None)
            available_plans.append("look around")
            actions.append(action)
        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans, actions

    def action_result_to_description(self, result, info):
        if result:
            return "success"
        elif info == 'max steps reached':
            return "paused after taking 100 steps"
        else:
            return f"fail, because {info}"

    def choose_target(self, state, processed_input):
        self.save_dir = processed_input['save_dir']
        agent_map = state["goal_map"]
        agent_points = (agent_map == -2).nonzero()
        if type(agent_points[0]) == np.ndarray:
            agent_pos = (agent_points[0].astype(float).mean(), agent_points[1].astype(float).mean())
        else:
            agent_pos = (agent_points[:, 0].float().mean().item(), agent_points[:, 1].float().mean().item())
        self.current_step = processed_input['step']
        holding_objects = processed_input['holding_objects']
        object_list = processed_input['explored_object_name_list']
        self.current_state = state
        self.current_seen_objects_id = [str(x) for x in list(set(self.current_state["raw"]['seg_mask'].flatten()))]
        if self.debug:
            print(f"current seen objects id: {self.current_seen_objects_id}")
        nearest_object = processed_input['nearest_object']
        action_result = processed_input['action_result']
        action_info = processed_input['action_info']
        self.update_object_status()
        if self.task == 'wind':
            if len(processed_input['holding_objects']) == 0 and \
                    len(processed_input['nearest_object']) > 0 and \
                    processed_input['nearest_object'][0]['category'] in self.target_objects:
                return "pick_up", processed_input['nearest_object'][0]['id']
            if len(processed_input['holding_objects']) > 0 and \
                    len(self.action_history) > 0 and \
                    self.action_history[-1].startswith('go put object into <shopping cart>') and \
                    processed_input['action_result']:
                return "drop", self.action_history[-1].split('(')[1].split(')')[0]
        else:
            if len(processed_input['holding_objects']) > 0 and \
                    processed_input['holding_objects'][0]['category'] in self.target_objects:
                self.picked_up_objects.append(int(processed_input['holding_objects'][0]['id']))
                return "drop", None
            if len(processed_input['nearest_object']) > 0 and \
                    processed_input['nearest_object'][0]['category'] in self.target_objects:
                return "pick_up", processed_input['nearest_object'][0]['id']

        self.update_history_action_result(action_result, action_info)
        action_history = [f"{action} ({self.action_result_to_description(result, info)})" for action, result, info in
                          zip(self.action_history, self.action_history_result, self.action_info_history)]
        action, info = self.run(self.current_step, holding_objects, nearest_object, object_list, action_history,
                                agent_pos)
        import json
        with open(os.path.join(self.save_dir, f'{self.current_step:04d}_info.json'), 'w') as f:
            json.dump(info, f, indent=4)
        return action

    def run(self, current_step, holding_objects, nearest_object, object_list, action_history, agent_pos):
        """Modified to execute the rescue plan"""
        
        self.update_state(current_step, holding_objects, nearest_object, object_list, agent_pos)
        prompt = ""
        
        # Generate new plan if needed
        if 0 == len(self.rescue_plan) or self.current_plan_index == len(self.rescue_plan) or self.current_plan_index >= self.max_plan_idx:
            print("Generating new rescue plan...")
            self.rescue_plan, prompt = self.plan_rescue_sequence()
            print(f"New rescue plan: {self.rescue_plan}")
            self.current_plan_index = 0
        
        # Get next target from plan
        if self.current_plan_index < len(self.rescue_plan):
            target_id = self.rescue_plan[self.current_plan_index]
            print("current target id:", target_id)
            print("current plan index:", self.current_plan_index)
            # Generate action to move toward/interact with current target
            action = self.get_action_for_target(target_id)
            
            # Update plan progress
            if self.target_reached(target_id):
                self.current_plan_index += 1
            
            return action, {"plan": self.rescue_plan, "current_target": target_id, "prompt": prompt}
        
        return ("explore", None), {"plan": [], "current_target": None, "prompt": prompt}

    def plan_rescue_sequence(self):
        """
        Plan the complete sequence of objects to rescue based on:
        - Object values
        - A* distances between objects
        - Current object states
        Returns a list of object IDs in optimal rescue order
        """
        object_edges = self.get_object_to_object_edges()
        progress_desc = self.progress2text()
        
        prompt = f"""Given the current state: {progress_desc}

        And the distances between objects:
        {self.format_object_to_object_edges()}

        Plan the optimal sequence to rescue all remaining valuable objects. Consider:
        1. Object values and priorities
        2. Distance between objects
        3. Current object conditions
        4. Most efficient path to collect objects

        Output ONLY the rescue plan as a sequence of object IDs, separated by commas, with no words, no explanations.
        Example format: 1, 2, 41, 12, 3
        """
        try:
            outputs, usage = self.generator([{"role": "user", "content": prompt}] if self.chat else prompt,
                                            self.sampling_params)
            
            plan_text = outputs[0]
            print(f"Generated rescue plan text: {plan_text}")
            
            object_sequence = list(map(int, re.findall(r"\d+", plan_text)))

            self.rescue_plan = object_sequence
            self.current_plan_index = 0
            self.plan_complete = False
            return object_sequence, prompt
            
        except Exception as e:
            print(f"Error generating rescue plan: {e}")
            return [], prompt

    def get_action_for_target(self, target_id):
        """
        Generate the action needed to move toward or interact with the target object.
        """
        target_objs = [obj for obj in self.object_list if obj['category'] in self.target_objects]
        obj_ids = [int(obj['id']) for obj in target_objs]
        for id in obj_ids:
            if int(id) == int(target_id):
                return ("walk_to", target_id)
        return ("explore", None)

    def target_reached(self, target_id):
        """
        Check if the agent has reached and picked up/dropped the target object.
        """
        print("Picked up objects:", self.picked_up_objects)
        if int(target_id) in self.picked_up_objects:
            return True
        return False

    def update_state(self, current_step, holding_objects, nearest_object, object_list, agent_pos):
        """
        Update the internal state of the agent, including its position, held objects, and the current step.
        """
        self.current_step = current_step
        self.holding_objects = holding_objects
        self.nearest_object = nearest_object
        self.object_list = object_list
        self.agent_pos = agent_pos

    def get_object_center(self):
        object_center_list = []
        id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
        object_ids = [int(obj['id']) for obj in self.object_list]
        # list of object IDs

        # agent position in real + grid space
        agent_grid = self.agent_pos
        if not isinstance(agent_grid, (list, tuple, np.ndarray)) or len(agent_grid) < 2:
            print("[A*] ERROR: self.agent_pos malformed:", agent_grid)
            return ["planner error"]

        # ensure ints
        agent_grid = (int(round(agent_grid[0])), int(round(agent_grid[1])))

        for idx in object_ids:
            # object pixels in semantic map
            object_points = (id_map == idx).nonzero()

            # compute object center in grid coordinates
            if type(object_points[0]) == np.ndarray:  # numpy
                center = (object_points[0].astype(float).mean(),
                        object_points[1].astype(float).mean())
                object_center_list.append((round(center[0], 2), round(center[1], 2)))
        
        return object_center_list
    
    def get_astar_distance(data, from_id, to_id):
        """
        Returns the A* path length between two nodes (agent or object) by their IDs.
        from_id: "agent" or object id (int)
        to_id: object id (int)
        Returns: (distance, reachable) or (None, False) if not found
        """
        if from_id == "agent":
            # Search agent_to_objects edges
            for edge in data["edges"]["agent_to_objects"]:
                if edge["to"] == to_id:
                    return edge["astar_cost"], edge["reachable"]
            return None, False
        else:
            # Search object_to_object edges
            for edge in data["edges"]["object_to_object"]:
                if edge["from"] == from_id and edge["to"] == to_id:
                    return edge["astar_cost"], edge["reachable"]
                if edge["from"] == to_id and edge["to"] == from_id:
                    return edge["astar_cost"], edge["reachable"]
            return None, False

    def get_object_to_object_edges(self):
        """
        Returns a list of dicts with A* distances between all unique pairs (combinations, not permutations)
        of target objects in self.object_list.
        Example:
        [
          {"from": 12, "to": 37, "astar_cost": 45.3, "reachable": True},
          ...
        ]
        """
        edges = []
        # Only target objects
        target_objs = [obj for obj in self.object_list if obj['category'] in self.target_objects]
        id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
        obj_ids = [int(obj['id']) for obj in target_objs]

        for i in range(len(obj_ids)):
            from_id = obj_ids[i]
            # Get center of the source object
            from_points = (id_map == from_id).nonzero()
            if type(from_points[0]) == np.ndarray:
                from_center = (from_points[0].astype(float).mean(), from_points[1].astype(float).mean())
            else:
                from_center = (from_points[:, 0].float().mean().item(), from_points[:, 1].float().mean().item())
            from_grid = (int(round(from_center[0])), int(round(from_center[1])))

            for j in range(i + 1, len(obj_ids)):
                to_id = obj_ids[j]
                to_points = (id_map == to_id).nonzero()
                if type(to_points[0]) == np.ndarray:
                    to_center = (to_points[0].astype(float).mean(), to_points[1].astype(float).mean())
                else:
                    to_center = (to_points[:, 0].float().mean().item(), to_points[:, 1].float().mean().item())
                to_grid = (int(round(to_center[0])), int(round(to_center[1])))

                try:
                    weight = get_astar_weight(sem_map=self.current_state['sem_map'],
                                             origin=from_grid,
                                             destination=to_grid)
                    path = get_astar_path(weight=weight,
                                          origin=from_grid,
                                          destination=to_grid)
                    if path is not None and len(path) > 1:
                        dist = 0.0
                        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                            dx, dy = abs(x2 - x1), abs(y2 - y1)
                            dist += np.sqrt(2) if (dx == 1 and dy == 1) else 1.0
                        edges.append({"from": from_id, "to": to_id, "astar_cost": round(dist, 2), "reachable": True})
                    else:
                        edges.append({"from": from_id, "to": to_id, "astar_cost": None, "reachable": False})
                except Exception:
                    edges.append({"from": from_id, "to": to_id, "astar_cost": None, "reachable": False})
        return edges
    
    def format_object_to_object_edges(self):
        """
        Returns a string describing object-to-object A* distances for the prompt.
        """
        edges = self.get_object_to_object_edges()
        if not edges:
            return ""
        s = "Object-to-object A* distances:\n"
        for edge in edges:
            if edge["reachable"]:
                s += f"from {edge['from']} to {edge['to']}: {edge['astar_cost']} (reachable)\n"
            else:
                s += f"from {edge['from']} to {edge['to']}: unreachable\n"
        return s