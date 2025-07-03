import os
import json
from pathlib import Path
from commandParser import CommandParser
from buildGraph import build_graph_from_trajectory


def batch_generate_graphs(trajs_file):
    eval_report_path = os.path.join(
        "/".join(trajs_file.split("/")[:-1]),
        "report.json"
    )
    patch_metrics_path = os.path.join(
        "/".join(trajs_file.split("/")[:-1]),
        "patch_metrics.jsonl"
    )
    graph_dir = "/".join(trajs_file.split("/")[:-1]).replace("evaluation/evaluation_outputs/outputs/", "graphs/")
    Path(graph_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(eval_report_path):
        print(f"Evaluation report not found at {eval_report_path}. Exiting.")
        sys.exit(1)

    parser = CommandParser()
    with open(trajs_file) as f:
        for idx, line in enumerate(f):
            traj = json.loads(line)
            if not traj:
                continue
            print(f"{idx + 1}: Processing trajectory with ID {traj.get('instance_id', 'unknown')}")
            build_graph_from_trajectory(traj, parser, graph_dir, eval_report_path, patch_metrics_path)

if __name__ == "__main__":
    trajs_file = "../../OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/output.jsonl"
    batch_generate_graphs(trajs_file)
    