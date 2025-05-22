import os
from datasets import load_dataset

PATH_TO_TRAJ = "trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test"

def count_invalid_instance(root_path):
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    instance_ids = [entry["instance_id"] for entry in dataset]

    # Map instance_id to its index in the dataset
    instance_id_to_index = {instance_id: idx for idx, instance_id in enumerate(instance_ids)}
    # print(instance_id_to_index)
    
    subdirs = [
        name for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    # Identify missing instance_ids (issues not present as subdirectories)
    missing_instance_ids = sorted(set(instance_ids) - set(subdirs))

    # Identify subdirectories without a .traj file
    subdirs_without_traj = sorted([
        name for name in subdirs
        if not any(
            fname.endswith(".traj")
            for fname in os.listdir(os.path.join(root_path, name))
            if os.path.isfile(os.path.join(root_path, name, fname))
        )
    ])

    # Output results
    print("\nMissing instance_ids:")
    for instance_id in missing_instance_ids:
        print(f"  {instance_id}")

    print("\nInstances without a .traj file:")
    for subdir in subdirs_without_traj:
        print(f"  {subdir}")

    # Report the index positions of missing instance_ids
    print("\nIndex positions of error instance_ids in the dataset:")
    for instance_id in (missing_instance_ids + subdirs_without_traj):
        index = instance_id_to_index[instance_id]
        print(f"  {instance_id}: Index {index}")

    # # Remove directories without .traj
    # print("\nRemoving subdirectories without .traj...")
    # for d in subdirs_without_traj:
    #     dir_to_remove = os.path.join(root_path, d)
    #     try:
    #         shutil.rmtree(dir_to_remove)
    #         print(f"  Removed: {d}")
    #     except Exception as e:
    #         print(f"  Failed to remove {d}: {e}")

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Remove subdirectories that do not contain a .traj file.")
    # parser.add_argument("path", type=str, help="Path to the root directory.")
    # args = parser.parse_args()

    # count_invalid_instance(args.path)
    count_invalid_instance(PATH_TO_TRAJ)

