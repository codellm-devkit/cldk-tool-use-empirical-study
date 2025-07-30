import importlib.util
from collections import Counter
import matplotlib.pyplot as plt
import csv

def load_module_from_path(path, module_name="swebench_constants"):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def extract_python_versions(file_path):
    mod = load_module_from_path(file_path)
    version_mapping = {}  # (project==version) -> python version
    structured_mapping = []  # list of tuples: (project, version, python_version)

    map_dict = mod.MAP_REPO_VERSION_TO_SPECS_PY
    for repo, version_spec_dict in map_dict.items():
        for version, spec in version_spec_dict.items():
            python_version = spec.get("python")
            if python_version:
                key = f"{repo}=={version}"
                version_mapping[key] = python_version
                structured_mapping.append((repo, version, python_version))

    return version_mapping, structured_mapping

def plot_distribution(version_mapping):
    counter = Counter(version_mapping.values())
    
    def parse_version(v):
        return tuple(map(int, v.split(".")))

    sorted_items = sorted(counter.items(), key=lambda x: parse_version(x[0]))
    labels, counts = zip(*sorted_items)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel("Python Version")
    plt.ylabel("Number of Project Versions")
    plt.title("Distribution of Python Versions Across SWE-Bench Instances")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("swebench_python_version_distribution.png")

def save_mapping_csv(structured_mapping, csv_path="swebench_python_versions.csv"):
    # Sort by (project, version)
    structured_mapping.sort(key=lambda x: (x[0], x[1]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Project", "Version", "Python Version"])
        writer.writerows(structured_mapping)

if __name__ == "__main__":
    constant_path = "./swebench_constant.py"  # update if needed
    version_map, structured_map = extract_python_versions(constant_path)

    plot_distribution(version_map)
    save_mapping_csv(structured_map)
