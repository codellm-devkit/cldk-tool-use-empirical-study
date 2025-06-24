import os
import json
import tempfile

ERROR_MSG = ('litellm.BadRequestError: DeepseekException - '
             '{"error":{"message":"Insufficient Balance","type":"unknown_error",'
             '"param":null,"code":"invalid_request_error"}}')

def find_error_instances(logs_dir):
    error_ids = set()
    for entry in os.listdir(logs_dir):
        if entry.startswith("instance_") and entry.endswith(".log"):
            inst = entry[len("instance_"):-len(".log")]
            path = os.path.join(logs_dir, entry)
            try:
                with open(path, encoding='utf-8') as f:
                    for line in f:
                        if ERROR_MSG in line:
                            error_ids.add(inst)
                            break
            except (OSError, UnicodeDecodeError) as e:
                print(f"Warning: skipping {entry}: {e}")
    print(f"Identified {len(error_ids)} instances with Insufficient Balance.")
    return error_ids

def remove_instances_from_jsonl(jsonl_path, error_ids):
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="filtered_", suffix=".jsonl")
    os.close(tmp_fd)
    kept = 0
    removed = 0
    with open(jsonl_path, encoding='utf-8') as src, open(tmp_path, 'w', encoding='utf-8') as dst:
        for line in src:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
            inst = rec.get("instance_id")
            if inst in error_ids:
                removed += 1
                continue
            dst.write(line)
            kept += 1
    os.replace(tmp_path, jsonl_path)
    print(f"Removed {removed} records; kept {kept} records in '{jsonl_path}'.")

def main():
    logs_dir = "/home/shuyang/cldk-tool-use-empirical-study/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/infer_logs"
    jsonl = logs_dir.replace("infer_logs", "output.jsonl")
    error_ids = find_error_instances(logs_dir)
    if error_ids:
        remove_instances_from_jsonl(jsonl, error_ids)
    else:
        print("No error instances found; output.jsonl remains unchanged.")

if __name__ == "__main__":
    main()
