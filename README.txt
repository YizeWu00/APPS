APPS 2.0
1.Run apps_create_split.py to get a data split.
    python apps_create_split.py \
        -r /path/to/APPS
    You will have "train.jsonl" and "test.jsonl" in the APPS root directory.
    They are data splits. "test.jsonl" will be used in the evaluation process.
    In apps_create_split.py line 19, you can CUSTOMIZE data split as ("split_name",number_of_tasks_in_this_split), e.g. ("validate", 1000),
        and there will be a "{split_name}.jsonl" file in the APPS root.
    Of course, you can otherwise create new train.jsonl or test.jsonl file(s) directly.

2.Run example_generation.py to get an example of code results.
    python example_generation.py \
        -t /path/to/test.jsonl \
        -r /path/to/APPS \
        --target /path/to/code/result.jsonl \
        -k number of generations per task
    The target file (results of codes) can be anywhere, and can be any name.
    The default value of k is 1. If you wanna generate more than 1 codes per task, use -k.
    As in example, generation result SHOULD BE [{task_id:"test/0000", completion:"def code():..."}, ...]
    
3.Run test_one_solution.py to evaluate the code results.
    python test_one_solution.py \
        -t /path/to/test.jsonl\
        -r /path/to/APPS\
        -p \ # if you want to print result
        --source /path/to/code/result.jsonl \ 
        --target /path/to/evaluation/result.jsonl \
        --tmp-dir /path/to/tmp/dir \
        --delete \ # if you want to delete tmp dir after evaluation
        -l /path/to/log \
    The target file (results of evaluation) can be anywhere.
    If -p, the result will be printed. In test_one_solution.py line 138, you can change k_list.
    The tmp dir can be anywhere.
    
4.Run print_results.py to print result of evaluations
    python print_results.py \
        --source /path/to/evaluation/result.jsonl
    Results include: pass@k (k_list can be modified in file)
                     unittest pass rate and error rate.

NOTE:
    1. train data is different from test data. Be careful when splitting train data into test set.
ENV:
    ./conda_APPS.yml
