#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def extract_instance_ids(jsonl_path: Path):
    seen = set()
    for line in jsonl_path.open("r", encoding="utf-8"):
        try:
            obj = json.loads(line)
            iid = obj.get("instance_id")
            if iid:
                seen.add(iid)
        except json.JSONDecodeError:
            continue
    return sorted(seen)

def write_toml_snippet(iids, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        f.write("selected_ids = [\n")
        for iid in iids:
            f.write(f'  "{iid}",\n')
        f.write("]\n")
    print(f"Wrote {len(iids)} ids to {output_path}")

def write_plain_list(iids, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for iid in iids:
            f.write(f"{iid}\n")
    print(f"Wrote plain instance_id list to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate selected_ids TOML snippet from OpenHands output")
    parser.add_argument("jsonl", type=Path, help="Path to output.jsonl")
    parser.add_argument("--toml", type=Path, default=Path("selected_ids.toml"), help="Path to write TOML snippet")
    parser.add_argument("--list", type=Path, default=Path("completed_ids.txt"), help="Optional: plain list")
    args = parser.parse_args()

    ids = extract_instance_ids(args.jsonl)
    write_toml_snippet(ids, args.toml)
    write_plain_list(ids, args.list)
