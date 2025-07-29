import json
from datasets import load_dataset
from swebench.versioning.get_versions import get_version

def extract_versions(dataset_name="princeton-nlp/SWE-bench_Verified", split="test"):
    # Load SWE-bench Verified dataset
    dataset = load_dataset(dataset_name, split=split)

    instance_versions = []

    for instance in dataset:
        try:
            version = get_version(instance, is_build=True)
        except Exception as e:
            version = "ERROR"
            print(f"Failed to get version for {instance['instance_id']}: {e}")

        instance_versions.append({
            "instance_id": instance["instance_id"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "version": version
        })

    return instance_versions

def main():
    versions = extract_versions()

    # Save to a JSONL file
    with open("swebench_verified_versions.jsonl", "w") as f:
        for item in versions:
            f.write(json.dumps(item) + "\n")

    print(f"Extracted versions for {len(versions)} instances.")

if __name__ == "__main__":
    main()
