import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
import datetime
from functions.utils import *

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class ConstraintGenerator:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            base_url='https://xiaoai.plus/v1',
            api_key=self.config['api_key']
        )
        
        # self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vlm_query')
        self.base_dir = 'vlm_query'  
        with open(os.path.join(self.base_dir, 'prompt_template_dualarm.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        prompt_text = self.prompt_template.format(instruction=instruction)
        # save prompt
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template.format(instruction=instruction)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages

    def _parse_and_save_constraints(self, output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                name = line.split("(")[0].split("def ")[1]
            if line.startswith("    return "):
                end = i
                functions[name] = lines[start:end+1]
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the constraint idx
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")
    

    def _parse_other_metadata_dualarm(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])

        # find grasp_keypoints_robot1
        grasp_keypoints_robot1_template = "grasp_keypoints_robot1 = {grasp_keypoints_robot1}"
        for line in output.split("\n"):
            grasp_keypoints_robot1 = parse.parse(grasp_keypoints_robot1_template, line)
            if grasp_keypoints_robot1 is not None:
                break
        if grasp_keypoints_robot1 is None:
            raise ValueError("grasp_keypoints_robot1 not found in output")
        # convert into list of ints
        grasp_keypoints_robot1 = grasp_keypoints_robot1['grasp_keypoints_robot1'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints_robot1 = [int(x.strip()) for x in grasp_keypoints_robot1]
        data_dict['grasp_keypoints_robot1'] = grasp_keypoints_robot1

        # find grasp_keypoints_robot2
        grasp_keypoints_robot2_template = "grasp_keypoints_robot2 = {grasp_keypoints_robot2}"
        for line in output.split("\n"):
            grasp_keypoints_robot2 = parse.parse(grasp_keypoints_robot2_template, line)
            if grasp_keypoints_robot2 is not None:
                break
        if grasp_keypoints_robot2 is None:
            raise ValueError("grasp_keypoints_robot2 not found in output")
        # convert into list of ints
        grasp_keypoints_robot2 = grasp_keypoints_robot2['grasp_keypoints_robot2'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints_robot2 = [int(x.strip()) for x in grasp_keypoints_robot2]
        data_dict['grasp_keypoints_robot2'] = grasp_keypoints_robot2

        # find release_keypoints_robot1
        release_keypoints_robot1_template = "release_keypoints_robot1 = {release_keypoints_robot1}"
        for line in output.split("\n"):
            release_keypoints_robot1 = parse.parse(release_keypoints_robot1_template, line)
            if release_keypoints_robot1 is not None:
                break
        if release_keypoints_robot1 is None:
            raise ValueError("release_keypoints_robot1 not found in output")
        # convert into list of ints
        release_keypoints_robot1 = release_keypoints_robot1['release_keypoints_robot1'].replace("[", "").replace("]", "").split(",")
        release_keypoints_robot1 = [int(x.strip()) for x in release_keypoints_robot1]
        data_dict['release_keypoints_robot1'] = release_keypoints_robot1

        # find release_keypoints_robot2
        release_keypoints_robot2_template = "release_keypoints_robot2 = {release_keypoints_robot2}"
        for line in output.split("\n"):
            release_keypoints_robot2 = parse.parse(release_keypoints_robot2_template, line)
            if release_keypoints_robot2 is not None:
                break
        if release_keypoints_robot2 is None:
            raise ValueError("release_keypoints_robot2 not found in output")
        # convert into list of ints
        release_keypoints_robot2 = release_keypoints_robot2['release_keypoints_robot2'].replace("[", "").replace("]", "").split(",")
        release_keypoints_robot2 = [int(x.strip()) for x in release_keypoints_robot2]
        data_dict['release_keypoints_robot2'] = release_keypoints_robot2

        return data_dict


    def _parse_other_metadata(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints 
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        # convert into list of ints
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        release_keypoints = [int(x.strip()) for x in release_keypoints]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate(self, img, instruction, metadata):
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        # create a directory for the task
        now = datetime.datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        date_folder_path = os.path.join(self.base_dir, date_folder)
        os.makedirs(date_folder_path, exist_ok=True)
        time_folder = now.strftime("%H-%M-%S")
        self.task_dir = os.path.join(date_folder_path, time_folder)
        os.makedirs(self.task_dir, exist_ok=True)
        # fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
        # self.task_dir = os.path.join(self.base_dir, fname)
        # os.makedirs(self.task_dir, exist_ok=True)
        
        # save query image
        image_path = os.path.join(self.task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        # build prompt
        messages = self._build_prompt(image_path, instruction)
        # stream back the response
        print(self.config['model'])
        stream = self.client.chat.completions.create(model=self.config['model'],
                                                        messages=messages,
                                                        temperature=self.config['temperature'],
                                                        max_tokens=self.config['max_tokens'],
                                                        stream=True)
        print("stream is created")
        output = ""
        start = time.time()
        for chunk in stream:
            if not chunk.choices:
                print("No choices in chunk.")
                continue
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
        # save raw output
        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        # parse and save constraints
        self._parse_and_save_constraints(output, self.task_dir)
        # save metadata
        # metadata.update(self._parse_other_metadata(output))
        metadata.update(self._parse_other_metadata_dualarm(output))
        self._save_metadata(metadata)
        return self.task_dir

if __name__ == "__main__":
    global_config = get_config(config_path="/home/luo/ICRA/configs/config.yaml")
    candidate_keypoints = [
        [-0.10052472352981567, 0.03343735262751579, 0.29600000381469727],
        [-0.054263293743133545, -0.019671978428959846, 0.2110000103712082],
        [-0.010573464445769787, -0.012453624978661537, 0.32100000977516174],
        [0.09238336235284805, -0.02311277762055397, 0.25699999928474426]
    ]
    save_path = '/home/luo/ICRA/vlm_query/cup'
    image_path = os.path.join(save_path, "query_img.png")
    loaded_image = cv2.imread(image_path)
    loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    projected = loaded_image

    metadata = {'init_keypoint_positions': candidate_keypoints, 'num_keypoints': len(candidate_keypoints)}
    instruction = 'Pick up the lid of the white cup and place it on top of the cube.'
    constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
    ICRA_program_dir = constraint_generator.generate(projected, instruction, metadata)


