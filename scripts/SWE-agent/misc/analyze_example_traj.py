import json
import os
import tiktoken

def tokenize(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text or ""))

def analyze_first_actions(traj_path, output_path, num_steps=1):
    try:
        with open(traj_path, 'r') as f:
            data = json.load(f)
            trajectory = data.get("trajectory", [])
            first_steps = trajectory[0:num_steps]

            analyzed = []
            total_execution_time = 0
            total_tokens_sent = 0
            total_tokens_received = 0

            for step in first_steps:
                action_str = step.get("action", "").strip()
                thought = step.get("thought", "").strip()
                observation = step.get("observation", "").strip()
                execution_time = step.get("execution_time", None)
                process = step.get("process", None)

                if execution_time is not None:
                    total_execution_time += execution_time

                tokens_sent = tokenize(thought)
                tokens_received = tokenize(observation)

                total_tokens_sent += tokens_sent
                total_tokens_received += tokens_received

                analyzed.append({
                    "action": action_str,
                    "thought": thought,
                    "execution_time": execution_time,
                    "process": process,
                    "tokens_sent": tokens_sent,
                    "tokens_received": tokens_received
                })

            result = {
                "steps": analyzed,
                "total_execution_time": total_execution_time,
                "total_tokens_sent": total_tokens_sent,
                "total_tokens_received": total_tokens_received
            }

            with open(output_path, 'w') as out_f:
                json.dump(result, out_f, indent=2)

    except Exception as e:
        print(f"Error processing {traj_path}: {e}")

if __name__ == "__main__":
    # before_traj_path = '/home/shuyang/cldk-tool-use-empirical-study/SWE-agent/trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/pydata__xarray-3305/pydata__xarray-3305.traj'
    # before_output_path = '../motivating_example/sweagent_anthropic_filemap__deepseek_pydata_xarray-3305.json'
    # if not os.path.exists(os.path.dirname(before_output_path)):
    #     os.makedirs(os.path.dirname(before_output_path))
    # analyze_first_actions(before_traj_path, before_output_path)
    # print(f"Analysis saved to {before_output_path}")

    after_traj_path = '/home/shuyang/cldk-tool-use-empirical-study/SWE-agent/trajectories/shuyang/structural_search__deepseek--deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/pydata__xarray-3305/pydata__xarray-3305.traj'
    after_output_path = '../motivating_example/sweagent_structural_searcher__deepseek_pydata_xarray-3305.json'
    if not os.path.exists(os.path.dirname(after_output_path)):
        os.makedirs(os.path.dirname(after_output_path))
    analyze_first_actions(after_traj_path, after_output_path)
    print(f"Analysis saved to {after_output_path}")