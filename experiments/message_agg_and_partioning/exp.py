import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import os
import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
from pregel import Pregel, Vertex, PageRankVertex, create_edges
from mods import AggregatedMessagePregel, PartitionedPregel
import gc
import csv
import pickle

NUM_WORKERS_LIST = [2, 4]
NUM_NODES_LIST = [1000, 5000]

execution_time_workers = {"original": [], "aggregated": [], "partitioned": []}
execution_time_nodes = {"original": [], "aggregated": [], "partitioned": []}
speedup_workers = {"original": [], "aggregated": [], "partitioned": []}
speedup_nodes = {"original": [], "aggregated": [], "partitioned": []}
load_balancing_workers = {"original": [], "aggregated": [], "partitioned": []}
load_balancing_nodes = {"original": [], "aggregated": [], "partitioned": []}

GRAPHS_DIR = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

def save_graph(vertices, filename):
    """Save the graph by storing only IDs instead of full objects."""
    graph_data = [(v.id, v.value, [neighbor.id for neighbor in v.out_vertices]) for v in vertices]
    with open(filename, "wb") as f:
        pickle.dump(graph_data, f)

def load_graph(filename):
    """Load the graph and reconstruct Vertex objects."""
    with open(filename, "rb") as f:
        graph_data = pickle.load(f)
    vertices = {vid: PageRankVertex(vid, value, []) for vid, value, _ in graph_data}
    for vid, _, neighbors in graph_data:
        vertices[vid].out_vertices = [vertices[nid] for nid in neighbors]
    return list(vertices.values())

def get_graph_filename(num_vertices):
    """Generate a unique filename based on the number of vertices."""
    return os.path.join(GRAPHS_DIR, f"graph_{num_vertices}.pkl")

def measure_pregel_performance(pregel_class, vertices_data, num_workers):
    """Runs a Pregel implementation and measures execution time & load balancing."""
    time.sleep(1)  
    graph_filename = get_graph_filename(len(vertices_data))

    graph_start_time = time.time()

    if os.path.exists(graph_filename):
        print(f"Loading graph from {graph_filename}... {len(vertices_data) = }\t{num_workers = }")
        vertices = load_graph(graph_filename)
    else:
        print(f"Generating new graph... {len(vertices_data) = }\t{num_workers = }")
        vertices = [PageRankVertex(vid, rank, []) for vid, rank in vertices_data]
        create_edges(vertices)
        save_graph(vertices, graph_filename)

    graph_end_time = time.time()

    print(f"Graph construction time: {graph_end_time - graph_start_time:.10f} seconds")
    pregel = pregel_class(vertices, num_workers)

    start_time = time.time()
    super_steps, messages = pregel.run()
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"Execution time: {total_time:.10f} seconds\t{pregel_class.__name__}\n")

    # Measure super steps, messages, and worker load balance
    worker_loads = [sum(len(v.out_vertices) for v in worker_list) for worker_list in pregel.partition.values()]
    std_dev_load = np.std(worker_loads)

    print(f"Super Steps: {super_steps}")
    print(f"Messages: {messages}")
    print(f"Standard Deviation of Worker Load: {std_dev_load}")
    
    return total_time, super_steps, messages, std_dev_load

def run_pregel_test(pregel_class, vertices_data, num_workers, return_dict, key):
    """Runs a Pregel test in a separate process and stores the result in return_dict."""
    time_taken, super_steps, messages, load = measure_pregel_performance(pregel_class, vertices_data, num_workers)
    return_dict[key] = (time_taken, super_steps, messages, load)

if __name__ == "__main__":  

    # run experiments for different worker sizes (keeping nodes fixed)
    fixed_num_nodes = 10000
    vertices_fixed = [(j, 1.0 / fixed_num_nodes) for j in range(fixed_num_nodes)]  

    for i, num_workers in enumerate(NUM_WORKERS_LIST):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        process_original = multiprocessing.Process(target=run_pregel_test, args=(Pregel, vertices_fixed, num_workers, return_dict, "original"))
        process_aggregated = multiprocessing.Process(target=run_pregel_test, args=(AggregatedMessagePregel, vertices_fixed, num_workers, return_dict, "aggregated"))
        process_partitioned = multiprocessing.Process(target=run_pregel_test, args=(PartitionedPregel, vertices_fixed, num_workers, return_dict, "partitioned"))

        process_original.start()
        process_aggregated.start()
        process_partitioned.start()

        process_original.join()
        process_aggregated.join()
        process_partitioned.join()

        original_time, original_steps, original_messages, original_load = return_dict["original"]
        aggregated_time, aggregated_steps, aggregated_messages, aggregated_load = return_dict["aggregated"]
        partitioned_time, partitioned_steps, partitioned_messages, partitioned_load = return_dict["partitioned"]

        execution_time_workers["original"].append(original_time)
        execution_time_workers["aggregated"].append(aggregated_time)
        execution_time_workers["partitioned"].append(partitioned_time)

        speedup_workers["original"].append(1.0)
        speedup_workers["aggregated"].append(original_time / aggregated_time if aggregated_time > 0 else float('inf'))
        speedup_workers["partitioned"].append(original_time / partitioned_time if partitioned_time > 0 else float('inf'))

        load_balancing_workers["original"].append(original_load)
        load_balancing_workers["aggregated"].append(aggregated_load)
        load_balancing_workers["partitioned"].append(partitioned_load)

        gc.collect()  

    # run experiments for different graph sizes (keeping workers fixed)
    fixed_num_workers = 6
    for i, num_nodes in enumerate(NUM_NODES_LIST):
        vertices_varied = [(j, 1.0 / num_nodes) for j in range(num_nodes)]  

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        process_original = multiprocessing.Process(target=run_pregel_test, args=(Pregel, vertices_varied, fixed_num_workers, return_dict, "original"))
        process_aggregated = multiprocessing.Process(target=run_pregel_test, args=(AggregatedMessagePregel, vertices_varied, fixed_num_workers, return_dict, "aggregated"))
        process_partitioned = multiprocessing.Process(target=run_pregel_test, args=(PartitionedPregel, vertices_varied, fixed_num_workers, return_dict, "partitioned"))

        process_original.start()
        process_aggregated.start()
        process_partitioned.start()

        process_original.join()
        process_aggregated.join()
        process_partitioned.join()

        original_time, original_steps, original_messages, original_load = return_dict["original"]
        aggregated_time, aggregated_steps, aggregated_messages, aggregated_load = return_dict["aggregated"]
        partitioned_time, partitioned_steps, partitioned_messages, partitioned_load = return_dict["partitioned"]

        execution_time_nodes["original"].append(original_time)
        execution_time_nodes["aggregated"].append(aggregated_time)
        execution_time_nodes["partitioned"].append(partitioned_time)

        speedup_nodes["original"].append(1.0)
        speedup_nodes["aggregated"].append(original_time / aggregated_time if aggregated_time > 0 else float('inf'))
        speedup_nodes["partitioned"].append(original_time / partitioned_time if partitioned_time > 0 else float('inf'))

        load_balancing_nodes["original"].append(original_load)
        load_balancing_nodes["aggregated"].append(aggregated_load)
        load_balancing_nodes["partitioned"].append(partitioned_load)

        gc.collect()  

    # Save results
    os.makedirs("results", exist_ok=True)
    csv_filename = f"results/pregel_experiment_results.csv"

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment Type", "Num Workers", "Num Nodes", "Original Time", "Aggregated Time", "Partitioned Time", 
                         "Original Speedup", "Aggregated Speedup", "Partitioned Speedup", 
                         "Original Load", "Aggregated Load", "Partitioned Load", 
                         "Original Super Steps", "Aggregated Super Steps", "Partitioned Super Steps", 
                         "Original Messages", "Aggregated Messages", "Partitioned Messages"])

        for i in range(len(NUM_WORKERS_LIST)):
            writer.writerow(["Workers", NUM_WORKERS_LIST[i], fixed_num_nodes, 
                             execution_time_workers["original"][i], execution_time_workers["aggregated"][i], execution_time_workers["partitioned"][i], 
                             speedup_workers["original"][i], speedup_workers["aggregated"][i], speedup_workers["partitioned"][i], 
                             load_balancing_workers["original"][i], load_balancing_workers["aggregated"][i], load_balancing_workers["partitioned"][i], 
                             original_steps, aggregated_steps, partitioned_steps, 
                             original_messages, aggregated_messages, partitioned_messages])

        for i in range(len(NUM_NODES_LIST)):
            writer.writerow(["Nodes", fixed_num_workers, NUM_NODES_LIST[i], 
                             execution_time_nodes["original"][i], execution_time_nodes["aggregated"][i], execution_time_nodes["partitioned"][i], 
                             speedup_nodes["original"][i], speedup_nodes["aggregated"][i], speedup_nodes["partitioned"][i], 
                             load_balancing_nodes["original"][i], load_balancing_nodes["aggregated"][i], load_balancing_nodes["partitioned"][i], 
                             original_steps, aggregated_steps, partitioned_steps, 
                             original_messages, aggregated_messages, partitioned_messages])

    print(f"Experiment results saved to {csv_filename}")

    # Plot results
    import subprocess
    try:
        subprocess.run(["python", "genplot.py", csv_filename], check=True)
        print("Plotting script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the plotting script: {e}")
