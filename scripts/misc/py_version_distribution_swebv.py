#!/usr/bin/env python3
import subprocess
import csv
import re
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

def list_swebench_env_images():
    """Return list of SWE‑bench Verified instance image tags."""
    out = subprocess.check_output(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"]
    ).decode()
    imgs = [ln.strip() for ln in out.splitlines()]
    # Filter by naming convention, e.g. "swebench/sweb.eval.*" 
    return [img for img in imgs if re.search(r"swebench\/sweb\.eval\.", img)]

def get_python_version(image_tag):
    """Run python --version inside the image and return version string."""
    try:
        out = subprocess.check_output(
            ["docker", "run", "--rm", image_tag, "python", "--version"],
            stderr=subprocess.STDOUT,
        ).decode().strip()
        return out.split()[-1]
    except subprocess.CalledProcessError:
        return "ERROR"

def main(csv_path="swebench_verified_python_versions.csv"):
    existing_mapping = {}

    # Load existing CSV if it exists
    csv_file = Path(csv_path)
    if csv_file.exists():
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_mapping[row["instance_image"]] = row["python_version"]

    updated = False
    for img in list_swebench_env_images():
        if img in existing_mapping:
            print(f"{img} already recorded, skipping.")
            continue
        ver = get_python_version(img)
        existing_mapping[img] = ver
        print(f"{img} -> {ver}")
        updated = True

    # Write updated mapping to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_image", "python_version"])
        for k, v in sorted(existing_mapping.items()):
            writer.writerow([k, v])

    # Plot distribution only if we have valid data
    versions = [ver for ver in existing_mapping.values() if ver != "ERROR"]
    cnt = Counter(versions)

    if cnt:
        labels, vals = zip(*cnt.most_common())
        plt.figure(figsize=(8, 6))
        plt.bar(labels, vals)
        plt.xticks(rotation=45)
        plt.xlabel("Python version")
        plt.ylabel("Number of instance images")
        plt.title("Python version distribution in SWE‑bench Verified images")
        plt.tight_layout()
        plt.savefig("python_version_distribution.png")
        print("Plot saved to python_version_distribution.png")
    else:
        print("No valid Python versions found; skipping plot.")

if __name__ == "__main__":
    main()
