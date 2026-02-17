# main.py
import argparse
import random
import torch
import numpy as np
from functions.main import Main
from functions.human_interaction import HumanInteraction
from functions.ICRA_simulation import ICRAsimulation
from functions.test_env_collision_avoidance import TestEnvCollisionAvoidance

"""
 _
| |     _   _   ___
| |    | | | | / _ \ 
| |___ | |_| || (_) |
|_____| \__,_| \___/

"""

if __name__ == "__main__":
    fix_seed = 2025
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='self-collision detection')
    parser.add_argument('--mode', type=str, choices=['rekep', 'human', 'test'], required=True, 
                        help='specify mode: rekep or human or test')
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--instruction', type=str, help='task text instruction')
    parser.add_argument('--enable_self_collision_avoidance', action='store_true', help='enable self-collision avoidance feature')
    parser.add_argument('--enable_env_collision_avoidance', action='store_true', help='enable env collision avoidance feature')    
    parser.add_argument('--human_pick', action='store_true', help='robot release the object and let human pick')
    parser.add_argument('--sim', action='store_true', help='Dual arm simulation')
    args = parser.parse_args()
    task_list = {
        'box': {
            'instruction': 'Pick up the black box and hold it 0.2m above the green pot, then place it at 5cm above the bottom of the pot (remember to release the box) and release the black box. When the tasks finish, robot end-effector returns to the safe position.',
            'rekep_program_dir': './vlm_query/box',
            },
        'toolbox': {
            'instruction': 'Pick up the blue box and hold it 0.2m above the green pot, then place it at 5cm above the bottom of the pot (z = 0.05m) and release the blue box. When the tasks finish, robot end-effector returns to the safe position.',
            'rekep_program_dir': './vlm_query/box',
            },
        'tool': {
            'instruction': 'Pick up the yellow tool and hold it 0.2m above the green pot, then place it at 5cm above the bottom of the pot (z = 0.05m) and release the yellow tool. When the tasks finish, robot end-effector returns to the safe position.',
            'rekep_program_dir': './vlm_query/box',
            },
        'pen': {
            'instruction': 'Pick up the black pen and put it in the pot.',
            'rekep_program_dir': './vlm_query/pen',
            },
        'dualarm': {
            'instruction': 'Pick up the black pen and put it in the pot.',
            'rekep_program_dir': './vlm_query/dualarm',
            },
        'surgery_tool': {
            'instruction': 'Robot 1 picks up the tweezers. Then Robot 1 holds it above 10 cm. Then Robot 1 releases the tweezers on the desk. Robot 2 keeps stll.',
            'rekep_program_dir': './vlm_query/surgery_tool',
            },
        'test': {
            'instruction': 'Robot 1 picks up the tweezers. Then Robot 1 holds it above 10 cm. Then Robot 1 releases the tweezers on the desk. Robot 2 keeps stll.',
            'rekep_program_dir': './vlm_query/test',
            },
    }
    task_name = args.task
    task = task_list.get(task_name)
    if args.instruction:
        instruction = args.instruction
    else:
        instruction = task['instruction']
    if args.mode == 'rekep':
        main = Main(args)
        main._perform_task(instruction,
                            rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None,
                            ask_gpt = not args.use_cached_query,
                        )
    elif args.mode == 'human':
        # human_interaction = HumanInteraction(args)
        # human_interaction._thread_start()

        ICRA_simulation = ICRAsimulation(args)
        ICRA_simulation._thread_start()

    elif args.mode == 'test':
        test_function = TestEnvCollisionAvoidance(args)
        test_function._thread_start()

    
    print("Done")