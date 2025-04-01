import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# csv_filename = "results/pregel_experiment_results_2.csv"
if len(sys.argv) < 2:
    print("Usage: python genplot.py <csv_filename>")
    sys.exit(1)

csv_filename = sys.argv[1]


num_workers, num_nodes = [], []
execution_time_workers, execution_time_nodes = {"original": [], "balanced": []}, {"original": [], "balanced": []}
speedup_workers, speedup_nodes = {"original": [], "balanced": []}, {"original": [], "balanced": []}
load_balancing_workers, load_balancing_nodes = {"original": [], "balanced": []}, {"original": [], "balanced": []}

with open(csv_filename, mode="r") as file:
    reader = csv.reader(file)
    next(reader)  

    for row in reader:
        exp_type, workers, nodes, orig_time, bal_time, orig_speedup, bal_speedup, orig_load, bal_load = row
        workers, nodes = int(workers), int(nodes)
        orig_time, bal_time = float(orig_time), float(bal_time)
        orig_speedup, bal_speedup = float(orig_speedup), float(bal_speedup)
        orig_load, bal_load = float(orig_load), float(bal_load)

        if exp_type == "Workers":
            num_workers.append(workers)
            execution_time_workers["original"].append(orig_time)
            execution_time_workers["balanced"].append(bal_time)
            speedup_workers["original"].append(orig_speedup)
            speedup_workers["balanced"].append(bal_speedup)
            load_balancing_workers["original"].append(orig_load)
            load_balancing_workers["balanced"].append(bal_load)
        else:
            num_nodes.append(nodes)
            execution_time_nodes["original"].append(orig_time)
            execution_time_nodes["balanced"].append(bal_time)
            speedup_nodes["original"].append(orig_speedup)
            speedup_nodes["balanced"].append(bal_speedup)
            load_balancing_nodes["original"].append(orig_load)
            load_balancing_nodes["balanced"].append(bal_load)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

axes[0, 0].bar(num_workers, execution_time_workers["original"], 
              width=0.4, label="Original", align="center")
axes[0, 0].bar([w + 0.4 for w in num_workers], 
              execution_time_workers["balanced"], 
              width=0.4, label="Balanced", align="center")
axes[0, 0].set_xticks([w + 0.2 for w in num_workers])  # Center ticks between bars
axes[0, 0].set_xticklabels(num_workers)
axes[0, 0].set_xlabel("Number of Workers")
axes[0, 0].set_ylabel("Execution Time (s)")
axes[0, 0].set_title("Execution Time vs Workers")
axes[0, 0].legend()

nodes_range = max(num_nodes) - min(num_nodes)
dynamic_width_nodes = nodes_range / (4 * len(num_nodes))  # adjust divisor for desired spacing

# execution time vs number of nodes
bar1 = axes[0, 1].bar(num_nodes, execution_time_nodes["original"], 
                     width=dynamic_width_nodes, label="Original", align="edge")
bar2 = axes[0, 1].bar([n + dynamic_width_nodes for n in num_nodes], 
                     execution_time_nodes["balanced"], 
                     width=dynamic_width_nodes, label="Balanced", align="edge")

axes[0, 1].set_xticks([n + dynamic_width_nodes/2 for n in num_nodes])
axes[0, 1].set_xticklabels(num_nodes)
axes[0, 1].set_xlabel("Number of Nodes")
axes[0, 1].set_ylabel("Execution Time (s)")
axes[0, 1].set_title("Execution Time vs Nodes")
axes[0, 1].legend()

# speedup vs number of workers 
axes[1, 0].plot(num_workers, speedup_workers["balanced"], marker='o', linestyle='-', color='tab:blue', label="Balanced")
for i, txt in enumerate(speedup_workers["balanced"]):
    axes[1, 0].annotate(f"{txt:.2f}", (num_workers[i], txt), textcoords="offset points", xytext=(0, 5), ha='center')
axes[1, 0].set_xlabel("Number of Workers")
axes[1, 0].set_ylabel("Speedup Factor")
axes[1, 0].set_title("Speedup vs Workers")
axes[1, 0].legend()

# speedup vs number of nodes 
axes[1, 1].plot(num_nodes, speedup_nodes["balanced"], marker='o', linestyle='-', color='tab:blue', label="Balanced")
for i, txt in enumerate(speedup_nodes["balanced"]):
    axes[1, 1].annotate(f"{txt:.2f}", (num_nodes[i], txt), textcoords="offset points", xytext=(0, 5), ha='center')
axes[1, 1].set_xlabel("Number of Nodes")
axes[1, 1].set_ylabel("Speedup Factor")
axes[1, 1].set_title("Speedup vs Nodes")
axes[1, 1].legend()

# load balancing vs number of workers
axes[2, 0].bar(num_workers, load_balancing_workers["original"], 
              width=0.4, label="Original", align="center")
axes[2, 0].bar([w + 0.4 for w in num_workers], 
              load_balancing_workers["balanced"], 
              width=0.4, label="Balanced", align="center")
axes[2, 0].set_xticks([w + 0.2 for w in num_workers])  # Center ticks between bars
axes[2, 0].set_xticklabels(num_workers)
axes[2, 0].set_xlabel("Number of Workers")
axes[2, 0].set_ylabel("Load Balancing (Std Dev)")
axes[2, 0].set_title("Load Balancing vs Workers")
axes[2, 0].legend()

# load balancing vs number of nodes
bar3 = axes[2, 1].bar(num_nodes, load_balancing_nodes["original"], 
                     width=dynamic_width_nodes, label="Original", align="edge")
bar4 = axes[2, 1].bar([n + dynamic_width_nodes for n in num_nodes], 
                     load_balancing_nodes["balanced"], 
                     width=dynamic_width_nodes, label="Balanced", align="edge")

axes[2, 1].set_xticks([n + dynamic_width_nodes/2 for n in num_nodes])
axes[2, 1].set_xticklabels(num_nodes)
axes[2, 1].set_xlabel("Number of Nodes")
axes[2, 1].set_ylabel("Load Balancing (Std Dev)")
axes[2, 1].set_title("Load Balancing vs Nodes")
axes[2, 1].legend()


plt.tight_layout()
import os
os.mkdir("results") if not os.path.exists("results") else None
plot_filename = f"{csv_filename[:-4]}.png"
plt.savefig(plot_filename)
# plt.show()
print(f"Plot: {plot_filename}")
# print(f"{execution_time_nodes = }")
# print(f"{load_balancing_nodes = }")