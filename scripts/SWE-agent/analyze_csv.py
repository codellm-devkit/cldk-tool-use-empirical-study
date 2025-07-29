import csv

input_file = '/home/shuyang/cldk-tool-use-empirical-study/SWE-agent/graphs/shuyang/analysis/trajectory_metrics.csv'
with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if int(row.get("fail_type_not found", 0)) > 0:
            print(row["instance"])
