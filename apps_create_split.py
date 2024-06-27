'''
Create data split for train and test (and more).
'''
import json
import os
import argparse
from apps_utils import write_jsonl

def main(args):
    # insert path to train and test
    # path should be relative to root directory
    paths_to_probs = ["train", "test"]
    # each problem is: {'task_id':'test/0000', 'path':'relative/path/from/root/to/problem', 'tag':'split tag'}
    all_problems = []
    for index, path_to_probs in enumerate(paths_to_probs):
        real_path_to_probs = os.path.join(args.root, path_to_probs)
        for _task_id in sorted(os.listdir(real_path_to_probs)):
            task_id = os.path.join(path_to_probs, _task_id)
            task_path = task_id
            all_problems.append(dict(task_id=task_id, path=task_path))
     
    splits = [("train", 5000), ("test", 5000)] # default
    count = 0
    for split in splits:
        if count >= len(all_problems):
            print(f"splitting {split[0]}: out of problems")
            break
        
        start = count
        count += split[1]
        end = count
        if count > len(all_problems):
            end = len(all_problems)

        # give tag
        for i in range(start, end):
            all_problems[i]['tag'] = split[0]
        
        # write this split to a jsonl
        split_jsonl_path = os.path.join(args.root, f"{split[0]}.jsonl")
        write_jsonl(split_jsonl_path, all_problems[start:end])
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data split")
    parser.add_argument("--root", "-r", default="./", type=str)
    
    args = parser.parse_args()
    main(args)