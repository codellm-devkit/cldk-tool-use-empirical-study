#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

def merge_preds(dir_path):
    dir_path = Path(dir_path)
    merged_preds = {}

    num_preds = 0
    # Iterate through all subdirectories
    for subdir in dir_path.iterdir():
        if subdir.is_dir():
            pred_file = subdir / f"{subdir.name}.pred"
            if pred_file.exists():
                with open(pred_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        instance_id = data.get("instance_id")
                        if instance_id:
                            num_preds += 1
                            merged_preds[instance_id] = data
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse {pred_file}: {e}")

    # Write to new_preds.json
    output_file = dir_path / "preds.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_preds, f, indent=2)
    
    print(f"Merged {num_preds} predictions into {output_file}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Merge .pred files into new_preds.json")
    # parser.add_argument("dir_path", type=str, help="Path to the directory containing .pred subdirectories")
    # args = parser.parse_args()

    # merge_preds(args.dir_path)
    dir_path = "/home/experiments/xvdc/cldk-tool-use-empirical-study/SWE-agent/trajectories/experiments/anthropic_filemap__GCP--claude-3-7-sonnet__t-0.00__p-1.00__c-2.00___swe_bench_verified_test"
    merge_preds(dir_path)
