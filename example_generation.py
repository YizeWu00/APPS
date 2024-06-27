"""
Example of generating Python code.
"""

import json
import random
import os
import pprint
import sys
import time
from apps_utils import read_problems, stream_jsonl, write_jsonl

# the format of the prompt can be customized
def generate_prompt(test_case_path, prompt_path, starter_path=None):
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"
    
    _input += "\nANSWER:\n"

    return _input


def generate_one_completion(prompts : list, solutions: list):
    return random.choice(solutions)        

def read_solutions(solutions_path: str):
    if os.path.isfile(solutions_path):
        with open(solutions_path, 'r') as f:
            sols = json.load(f)
    else:
        sols = ["print(\"hello world\")"]
    # sols = ["print(\"hello world\")"]
    return sols

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = read_problems(args.test_loc)
    # list of {task_id:xx, {"task_id": "test/0000", "path": "test/0000", "tag": "test"}}
    for task_id in problems:        
        prob_path = os.path.join(args.root, problems[task_id]['path'])
        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
            starter_path = None
        if not os.path.exists(test_case_path) or not os.path.exists(prompt_path):
            print(f"data corruption at test_path:{test_case_path}:{os.path.exists(test_case_path)}, prompt_path:{prompt_path}:{os.path.exists(prompt_path)}")
            continue

        # Read the question in
        problems[task_id]['prompt'] = generate_prompt(test_case_path, prompt_path, starter_path)
        # no need to read solutions, JUST FOR EXAMPLE USE!!!
        problems[task_id]['solutions'] = read_solutions(solutions_path)
        
    
    if not os.path.isdir(os.path.dirname(args.target)):
        os.makedirs(os.path.dirname(args.target), exist_ok=True)
        
    # generate code
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"], problems[task_id]['solutions']))
        for task_id in problems
        for _ in range(args.num_gen_per_task)
    ] # solutions JUST FOR EXAMPLE
    
    write_jsonl(args.target, samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Example generate Python code.")
    parser.add_argument("-t","--test_loc", default="./test.jsonl", type=str) # path to test.json file
    parser.add_argument("-r","--root", default="./", type=str, help="where the data is stored.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--target", type=str, default="./results/all_codes.jsonl")
    parser.add_argument("-k", "--num-gen-per-task", default=1, type=int)
 
    args = parser.parse_args()

    main(args)
