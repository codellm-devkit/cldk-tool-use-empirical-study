import os
import json
from gsppy.gsp import GSP

def aggregate_nodes_by_step_index(data):
    step_sequence = []

    for node in data.get("nodes", []):
        step_indices = node.get("step_indices", [])
        for idx in step_indices:
            step_sequence.append((idx, node))

    # Sort the list by step index and extract the node objects
    step_sequence.sort(key=lambda x: x[0])
    ordered_nodes = [node for _, node in step_sequence]
    return ordered_nodes

def extract_phase_sequence(ordered_nodes):
    phase_sequence = []
    prev_phase = None
    for node in ordered_nodes:
        curr_phase = node.get('phase')
        if curr_phase and curr_phase != "general" and curr_phase != prev_phase:
            # merges adjacent duplicate phases and skips general phases
            phase_sequence.append(curr_phase)
            prev_phase = curr_phase
    return phase_sequence

def extract_label_sequence(ordered_nodes):
    label_sequence = []
    for node in ordered_nodes:
        label = node.get('label', 'Unknown Node').replace('\n', ': ').strip()
        label_sequence.append(label)
    return label_sequence

if __name__ == "__main__":
    instance_dir = "graphs/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/"  

    phase_sequences = []
    label_sequences = []
    for root, _, files in os.walk(instance_dir):
        for fname in files:
            if fname.endswith(".json"):
                json_path = os.path.join(root, fname)
                with open(json_path, "r") as f:
                    data = json.load(f)

                print(f"\nProcessing: {fname}")
                ordered_nodes = aggregate_nodes_by_step_index(data)

                phase_sequence = extract_phase_sequence(ordered_nodes)
                label_sequence = extract_label_sequence(ordered_nodes)
                print("Phase Sequence: ", phase_sequence)
                # print("First 5 nodes: ", label_sequence[:5])
                phase_sequences.append(phase_sequence)
                label_sequences.append(label_sequence)
   
    # Find frequent patterns in phase sequences
    min_support = 0.3
    result = GSP(phase_sequences).search(min_support)
    print("\nFrequent Phase Patterns:")
    for pattern in result:
        print(pattern)