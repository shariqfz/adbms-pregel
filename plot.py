import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python genplot.py <csv_filename>")
    sys.exit(1)

csv_filename = sys.argv[1]

num_workers, num_nodes = [], []
execution_time_workers, execution_time_nodes = {"original": [], "balanced": []}, {"original": [], "balanced": []}
speedup_workers, speedup_nodes = {"original": [], "balanced": []}, {"original": [], "balanced": []}
load_balancing_workers, load_balancing_nodes = {"original": [], "balanced": []}, {"original": [], "balanced": []}
supersteps_workers, supersteps_nodes = {"original": [], "balanced": []}, {"original": [], "balanced": []}
messages_workers, messages_nodes = {"original": [], "balanced": []}, {"original": [], "balanced": []}

with open(csv_filename, mode="r") as file:
    reader = csv.reader(file)
    next(reader)  

    for row in reader:
        (exp_type, workers, nodes, orig_time, bal_time, orig_speedup, bal_speedup, orig_load, bal_load, 
         orig_supersteps, bal_supersteps, orig_messages, bal_messages) = row
        
        workers, nodes = int(workers), int(nodes)
        orig_time, bal_time = float(orig_time), float(bal_time)
        orig_speedup, bal_speedup = float(orig_speedup), float(bal_speedup)
        orig_load, bal_load = float(orig_load), float(bal_load)
        orig_supersteps, bal_supersteps = int(orig_supersteps), int(bal_supersteps)
        orig_messages, bal_messages = int(orig_messages), int(bal_messages)

        if exp_type == "Workers":
            num_workers.append(workers)
            execution_time_workers["original"].append(orig_time)
            execution_time_workers["balanced"].append(bal_time)
            speedup_workers["original"].append(orig_speedup)
            speedup_workers["balanced"].append(bal_speedup)
            load_balancing_workers["original"].append(orig_load)
            load_balancing_workers["balanced"].append(bal_load)
            supersteps_workers["original"].append(orig_supersteps)
            supersteps_workers["balanced"].append(bal_supersteps)
            messages_workers["original"].append(orig_messages)
            messages_workers["balanced"].append(bal_messages)
        else:
            num_nodes.append(nodes)
            execution_time_nodes["original"].append(orig_time)
            execution_time_nodes["balanced"].append(bal_time)
            speedup_nodes["original"].append(orig_speedup)
            speedup_nodes["balanced"].append(bal_speedup)
            load_balancing_nodes["original"].append(orig_load)
            load_balancing_nodes["balanced"].append(bal_load)
            supersteps_nodes["original"].append(orig_supersteps)
            supersteps_nodes["balanced"].append(bal_supersteps)
            messages_nodes["original"].append(orig_messages)
            messages_nodes["balanced"].append(bal_messages)

fig, axes = plt.subplots(4, 2, figsize=(14, 16))

# Execution Time vs Workers
axes[0, 0].bar(num_workers, execution_time_workers["original"], width=0.4, label="Original", align="center")
axes[0, 0].bar([w + 0.4 for w in num_workers], execution_time_workers["balanced"], width=0.4, label="Balanced", align="center")
axes[0, 0].set_xticks([w + 0.2 for w in num_workers])
axes[0, 0].set_xticklabels(num_workers)
axes[0, 0].set_xlabel("Number of Workers")
axes[0, 0].set_ylabel("Execution Time (s)")
axes[0, 0].set_title("Execution Time vs Workers")
axes[0, 0].legend()

# Execution Time vs Nodes
axes[0, 1].bar(num_nodes, execution_time_nodes["original"], width=0.4, label="Original", align="center")
axes[0, 1].bar([n + 0.4 for n in num_nodes], execution_time_nodes["balanced"], width=0.4, label="Balanced", align="center")
axes[0, 1].set_xticks([n + 0.2 for n in num_nodes])
axes[0, 1].set_xticklabels(num_nodes)
axes[0, 1].set_xlabel("Number of Nodes")
axes[0, 1].set_ylabel("Execution Time (s)")
axes[0, 1].set_title("Execution Time vs Nodes")
axes[0, 1].legend()

# Supersteps vs Workers
axes[1, 0].plot(num_workers, supersteps_workers["original"], marker='o', linestyle='-', label="Original")
axes[1, 0].plot(num_workers, supersteps_workers["balanced"], marker='o', linestyle='-', label="Balanced")
axes[1, 0].set_xlabel("Number of Workers")
axes[1, 0].set_ylabel("Supersteps")
axes[1, 0].set_title("Supersteps vs Workers")
axes[1, 0].legend()

# Supersteps vs Nodes
axes[1, 1].plot(num_nodes, supersteps_nodes["original"], marker='o', linestyle='-', label="Original")
axes[1, 1].plot(num_nodes, supersteps_nodes["balanced"], marker='o', linestyle='-', label="Balanced")
axes[1, 1].set_xlabel("Number of Nodes")
axes[1, 1].set_ylabel("Supersteps")
axes[1, 1].set_title("Supersteps vs Nodes")
axes[1, 1].legend()

# Messages vs Workers
axes[2, 0].plot(num_workers, messages_workers["original"], marker='o', linestyle='-', label="Original")
axes[2, 0].plot(num_workers, messages_workers["balanced"], marker='o', linestyle='-', label="Balanced")
axes[2, 0].set_xlabel("Number of Workers")
axes[2, 0].set_ylabel("Total Messages")
axes[2, 0].set_title("Messages vs Workers")
axes[2, 0].legend()

# Messages vs Nodes
axes[2, 1].plot(num_nodes, messages_nodes["original"], marker='o', linestyle='-', label="Original")
axes[2, 1].plot(num_nodes, messages_nodes["balanced"], marker='o', linestyle='-', label="Balanced")
axes[2, 1].set_xlabel("Number of Nodes")
axes[2, 1].set_ylabel("Total Messages")
axes[2, 1].set_title("Messages vs Nodes")
axes[2, 1].legend()

plt.tight_layout()
plt.savefig(f"{csv_filename[:-4]}.png")
print(f"Plot saved as {csv_filename[:-4]}.png")
