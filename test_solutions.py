"""
Run solutions from problem.
"""
import argparse
import os
import sys
import pprint
import multiprocessing
import warnings
import math
from collections import Counter

import testing_util as test_util
from print_results import print_results
from apps_utils import read_problems, stream_jsonl, write_jsonl

from tqdm import tqdm

# {'xx(id)':content} -> {'task_id':xx, 'item_name':item}, for JSON readable
def formatted(my_dict:dict, item_name:str):
    formatted = []
    for task_id in my_dict:
        formatted.append({'task_id':task_id, item_name:my_dict[task_id]})
    return formatted

def _temp_run(task_id, prob_path, generation, debug, o_idx, tmp_dir, multiprocess_results, multiprocess_logs):
    this_results, this_logs = test_util.run_test(prob_path=prob_path, test=generation, debug=debug, o_idx=o_idx, tmp_dir=tmp_dir)
    # manager.dict MUST use tmp to assign
    tmp = multiprocess_results[task_id]
    tmp[o_idx] = this_results
    multiprocess_results[task_id] = tmp
    tmp = multiprocess_logs[task_id]
    tmp[o_idx] = this_logs
    multiprocess_logs[task_id] = tmp

def check_correctness(task_id, prob_path, generation, debug, o_idx, tmp_dir, multiprocess_results, multiprocess_logs):
    """
    Check correctness of code generation.
    No global timeout.    
    """
    p = multiprocessing.Process(target=_temp_run, args=(task_id, prob_path, generation, debug, o_idx, tmp_dir, multiprocess_results, multiprocess_logs))
    return p


def eval_and_save_problems(args):

    if not os.path.isdir(os.path.dirname(args.target)):
        os.makedirs(os.path.dirname(args.target), exist_ok=True)
    if args.log and not os.path.isdir(os.path.dirname(args.log)):
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
    
    # {'(task_id)xx':{'task_id':xx, 'path':path, ...}}
    problems = read_problems(args.test_loc)
    # [{'task_id':xx, 'completion':code}]
    completions = [completion for completion in stream_jsonl(args.source)]
    # check if every task has a generation result
    problems_task_ids = list(problems.keys()); completions_task_ids = [completion['task_id'] for completion in completions] 
    if set(problems_task_ids) - set(completions_task_ids) != set():
        print(f"some problems are not solved: {set(problems_task_ids) - set(completions_task_ids)}")
        exit()
    elif set(completions_task_ids) - set(problems_task_ids) != set():
        print(f"some completions are not in problem list: {set(completions_task_ids) - set(problems_task_ids)}")
        exit()
    
    manager = multiprocessing.Manager()
    # {'xx(task_id)': [[true, -1, -2]]}
    multiprocess_results = manager.dict()
    # {'xx(task_id)': [["log content",...]]}
    multiprocess_logs = manager.dict()
    # stop SyntaxWarning
    warnings.simplefilter('ignore')
    
    # for multi process
    processes = []
    gen_idx = Counter()
    
    print("loading tests.")
    for completion in tqdm(completions):
        task_id = completion['task_id']
        output_str = completion['completion'] # a str of this completion
        prob_path = os.path.join(args.root, problems[task_id]['path'])
        
        if not multiprocess_results.get(task_id):
            multiprocess_results[task_id] = {}
            multiprocess_logs[task_id] = {}
        try:
            p = check_correctness(task_id=task_id, prob_path=prob_path, generation=output_str, 
                                  debug=args.debug, o_idx=gen_idx[task_id], tmp_dir=args.tmp_dir,
                                  multiprocess_results=multiprocess_results,
                                  multiprocess_logs=multiprocess_logs)
            processes.append(p)
            gen_idx[task_id] += 1
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            exit()
    
    print("start tests.")
    for p in tqdm(processes):
        p.start()
    print("wait tests to stop.")
    for p in tqdm(processes):
        p.join()
            
    def mp2sorted(src:dict,sort_func=None):
        '''
            src: DictProxy, {'xx(taskid)': ...}
            {'xx(task_id)': {'0(o_idx)':[-1,true], ...}} -> {'xx(task_id)': [[-1, true], ...]}
        '''
        target = {}
        if not sort_func:
            sort_func = sorted # default sort function
            
        for task_id in src:
            target[task_id] = []
            tmp = src[task_id] # DictProxy, tmp is a dict
            for key in sort_func(tmp):
                target[task_id].append(tmp[key])
        return target
    
    if args.log:
        try:
            write_jsonl(args.log, formatted(mp2sorted(multiprocess_logs), "logs"))
        except Exception as e:
            print(f"log write exception {e}")
            
    # formatted results for jsonl
    return formatted(mp2sorted(multiprocess_results), "results")


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    
    # make tmp dir
    if os.path.isdir(args.tmp_dir):
        print(f"{args.tmp_dir} exists. New temp dir is needed.")
        sys.exit()
    else:
        os.makedirs(args.tmp_dir)
        
    # add tmp_dir to sys.path, for tmp module
    # tmp_dir is in the execution dir, not the dataset dir
    # tmp_dir_path = os.path.join(os.getcwd(), "tmp")
    sys.path.append(args.tmp_dir)

    # evaluation
    results = eval_and_save_problems(args)
    
    # store result
    write_jsonl(args.target, results)
    
    # print result if needed
    if args.print_results:
        k_list = [1]
        print_results(results, k_list=k_list)
    # delete module file
    if args.delete:
        os.system(f"rm -r {args.tmp_dir}")

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t","--test_loc", default="./test.jsonl", type=str, help="path to the test config.")
    parser.add_argument("-r","--root", default="./", type=str, help="where the data is stored.")
    parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--source", type=str, default="./results/all_codes.jsonl", help="Where the code result is.")
    parser.add_argument("--target", type=str, default="./results/all_results.jsonl", help="Where the evalution result is stored.")
    parser.add_argument("--tmp-dir", default=os.path.join(os.getcwd(), "tmp"), type=str)
    parser.add_argument("--delete", action="store_true", help="Delete tmp file or not")
    
    parser.add_argument("-l", "--log", default=None, type=str)
 
    args = parser.parse_args()

    main(args)
