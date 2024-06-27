'''
    old test.json and all_codes.json -> new test.jsonl and all_codes.jsonl
'''
import json
from apps_utils import write_jsonl

old_test_loc = "./test1.json"
new_test_loc = "./test1.jsonl"
old_code_loc = "./results/all_codes1.json"
new_code_loc = "./results/all_codes1.jsonl"

with open(old_code_loc, "r") as f:
    old_code = json.load(f)
with open(old_test_loc, "r") as f:
    old_test = json.load(f)

new_test = []
new_code = []
for index in old_code:
    task_id = old_test[int(index)].replace("./", "")
    path = task_id
    completions = old_code[index]
    new_test.append(dict(task_id=task_id, path=path, tag="test"))
    for completion in completions:
        new_code.append(dict(task_id=task_id, completion=completion))

write_jsonl(new_test_loc, new_test)
write_jsonl(new_code_loc, new_code)
    