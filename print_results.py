'''
    Just print the evaluation result.
'''
import argparse
import numpy as np
from apps_utils import stream_jsonl
from testing_util import ERROR_TYPE

def print_results(results: list, k_list: list = [1]):
    """
    Given the results evaluated against the testcases, we output pass@k.
    @results: [{'task_id':xx, 'results':[[-1,-1], ...]}]
    @k_list: a list of int, for values of k

    """
    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    pass_at_k_res = []
    unit_correct = 0
    # global_timeout_error = 0
    unable_func_error = 0
    compile_error = 0
    timeout_error = 0
    failed_error = 0
    runtime_error = 0
    
    for res in results:
        # each task result: {'task_id':xx, 'results':[[-1, -2, true],...]}
        multi_code_one_task_res = res['results']
        n = len(multi_code_one_task_res)
        c = 0
        for one_code_one_task_res in multi_code_one_task_res:
            one_code_one_task_res = np.asarray(one_code_one_task_res)
            unable_func_error += np.sum(one_code_one_task_res == ERROR_TYPE.unable_func_error)
            timeout_error += np.sum(one_code_one_task_res == ERROR_TYPE.timeout_error)
            compile_error += np.sum(one_code_one_task_res == ERROR_TYPE.compile_error)
            runtime_error += np.sum(one_code_one_task_res == ERROR_TYPE.runtime_error)
            failed_error += np.sum(one_code_one_task_res == False)
            unit_correct += np.sum(one_code_one_task_res == True)
            if np.all(one_code_one_task_res == True):
                c += 1
        one_task_pass_at_k_res = []
        for k in k_list:
            one_task_pass_at_k_res.append(estimator(n, c, k))
        pass_at_k_res.append(one_task_pass_at_k_res)
    
    print(f"\nHow to read results: \
[{ERROR_TYPE.unable_func_error}] unable to get function error, \
[{ERROR_TYPE.timeout_error}] time exceeds limit error, \
[{ERROR_TYPE.compile_error}] = compile error, \
[{ERROR_TYPE.runtime_error}] = runtime error, \
[False] = failed test case, \
[True] = passed test case")
    
    # pass at k
    print(f"pass@k result is:")
    for i,k in enumerate(k_list):
        print(f"k = {k}, result = {np.mean(np.asarray(pass_at_k_res), axis=0)[i]}")
    # unit test
    total_unit_test = unable_func_error + timeout_error + compile_error + runtime_error + failed_error + unit_correct
    print(f"total unittest : {total_unit_test}")
    print(f"unable func [{ERROR_TYPE.unable_func_error}] = {unable_func_error}, {unable_func_error/total_unit_test}")
    print(f"timeout error [{ERROR_TYPE.timeout_error}] = {timeout_error}, {timeout_error/total_unit_test}")
    print(f"compile error [{ERROR_TYPE.compile_error}] = {compile_error}, {compile_error/total_unit_test}")
    print(f"runtime error [{ERROR_TYPE.runtime_error}] = {runtime_error}, {runtime_error/total_unit_test}")
    print(f"failed [False] = {failed_error}, {failed_error/total_unit_test}")
    print(f"passed [True] = {unit_correct}, {unit_correct/total_unit_test}")

def main(args):
    results = [res for res in stream_jsonl(args.source)]
    k_list = [1]
    print_results(results, k_list=k_list)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print Result")
    parser.add_argument("--source", type=str, default="./results/all_results.jsonl", help="Where the evaluated data saved to.")
    args = parser.parse_args()

    main(args)
