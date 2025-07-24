import os
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from datasets import load_dataset

PLOT_DIR = "plots"

def load_metrics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def summarize_file_count(metrics):
    file_count_counter = Counter(m["file_count"] for m in metrics)

    # Accurate counts
    single_file_instances = sum(1 for m in metrics if m["file_count"] == 1)
    multiple_file_instances = sum(1 for m in metrics if m["file_count"] > 1)

    print("File Count Breakdown:")
    print(f"  {'Single file':<15}: {single_file_instances}")
    print(f"  {'Multiple files':<15}: {multiple_file_instances}")
    print("  File Count Distribution:")     
    for count, freq in sorted(file_count_counter.items()):
        print(f"  {count:>2} files: {freq}")
    return file_count_counter

def summarize_difficulty(metrics):
    difficulty_counter = Counter(m["patch_difficulty"] for m in metrics)
    print("Patch Difficulty Breakdown:")
    for level in ["easy", "medium", "hard", "very hard"]:
        print(f"  {level:>9}: {difficulty_counter.get(level, 0)}")
    return difficulty_counter

def compare_with_swebench(metrics):
    print("\nComparing with SWE-bench developer-assigned difficulty...")
    swe_bench_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    difficulty_lookup = {
        row["instance_id"]: row["difficulty"] for row in swe_bench_ds
    }

    difficulty_rename = {
        "<15 min fix": "easy",
        "15 min - 1 hour": "medium",
        "1-4 hours": "hard",
        ">4 hours": "very hard"
    }

    levels = ["easy", "medium", "hard", "very hard"]
    confusion = defaultdict(lambda: defaultdict(int))
    dev_difficulty_counts = Counter()

    for m in metrics:
        inst_id = m["instance_id"]
        patch_diff = m.get("patch_difficulty")
        dev_diff_raw = difficulty_lookup.get(inst_id)
        dev_diff = difficulty_rename.get(dev_diff_raw, None)

        if dev_diff:
            dev_difficulty_counts[dev_diff] += 1
            if patch_diff:
                confusion[dev_diff][patch_diff] += 1

    print("\nConfusion Matrix (developer time vs patch complexity):")
    header = "dev_time \\ patch".ljust(18) + "  ".join(level.ljust(10) for level in levels)
    print(header)
    print("-" * len(header))
    for dev in levels:
        row = dev.ljust(18)
        total = 0
        for patch in levels:
            count = confusion[dev][patch]
            total += count
            row += str(count).ljust(10)
        print(row)

    total_instances = sum(dev_difficulty_counts.values())
    print(f"\nTotal instances: {total_instances}")


def plot_histogram(metrics, key: str, title: str, filename: str, bins: int = 30):
    values = [m[key] for m in metrics if key in m]
    if not values:
        print(f"[Warning] No data for: {key}")
        return
    plt.figure()
    plt.hist(values, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(key)
    plt.ylabel("Count")
    plt.grid(True)
    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")

def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)

if __name__ == "__main__":
    path = "../SWE-agent/trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/patch_metrics.jsonl"
    metrics = load_metrics(path)
    print(f"Loaded {len(metrics)} instances from SWE-agent (anthropic_filemap__deepseek)")
    summarize_file_count(metrics)

    path = "../OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/patch_metrics.jsonl"
    metrics = load_metrics(path)
    print(f"Loaded {len(metrics)} instances from OpenHands (deepseek)")
    summarize_file_count(metrics)

    path = "golden_patch_metrics.jsonl"  
    metrics = load_metrics(path)
    print(f"Loaded {len(metrics)} instances from {path}")
    summarize_file_count(metrics)
    # ensure_plot_dir()

    # summarize_difficulty(metrics)
    # compare_with_swebench(metrics)

    # plot_histogram(metrics, "difficulty_score", "Difficulty Score Distribution", "difficulty_score.png")
    # plot_histogram(metrics, "ABC_magnitude_sum", "ABC Magnitude Distribution", "abc_magnitude.png")
    # plot_histogram(metrics, "file_count", "Files per Patch", "file_count.png")
    # plot_histogram(metrics, "hunk_count", "Hunks per Patch", "hunk_count.png")
