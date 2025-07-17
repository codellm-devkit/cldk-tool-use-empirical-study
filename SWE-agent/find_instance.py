import sys
from datasets import load_dataset

if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <instance_id>")
# pydata__xarray-3305
target_id = sys.argv[1]

# Ensure you have run `huggingface-cli login` if the dataset requires authentication.
dataset = load_dataset("princeton-nlp/SWE-bench_Verified")

for split_name, split in dataset.items():
    for idx, record in enumerate(split):
        if record["instance_id"] == target_id:
            print(f"Found '{target_id}' in split '{split_name}' at row index {idx}")
            sys.exit(0)

print(f"Instance '{target_id}' not found.")