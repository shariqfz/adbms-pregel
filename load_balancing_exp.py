import os
import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
from pregel import Pregel, Vertex, PageRankVertex, create_edges
from mods import BalancedPregel
import gc
import csv
import pickle

NUM_WORKERS_LIST = [2, 4, 6, 8]
NUM_NODES_LIST = [1000, 5000, 10000, 20000, 50000, 100000]

execution_time_workers = {"original": [], "balanced": []}
execution_time_nodes = {"original": [], "balanced": []}
speedup_workers = {"original": [], "balanced": []}
speedup_nodes = {"original": [], "balanced": []}
load_balancing_workers = {"original": [], "balanced": []}
load_balancing_nodes = {"original": [], "balanced": []}

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
        # reconstruct vertices from picklable format
        vertices = [PageRankVertex(vid, rank, []) for vid, rank in vertices_data]
        create_edges(vertices)
        save_graph(vertices, graph_filename)

    graph_end_time = time.time()

    print(f"Graph construction time: {graph_end_time - graph_start_time:.10f} seconds")
    pregel = pregel_class(vertices, num_workers)

    start_time = time.time()
    supersteps, total_messages = pregel.run()
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"Execution time: {total_time:.5f}s | Supersteps: {supersteps} | Messages: {total_messages} | \t{pregel_class.__name__}\n")

    # worker load balance is (sum of vertex degrees per worker)
    worker_loads = [sum(len(v.out_vertices) for v in worker_list) for worker_list in pregel.partition.values()]
    std_dev_load = np.std(worker_loads)

    return total_time, std_dev_load

def run_pregel_test(pregel_class, vertices_data, num_workers, return_dict, key):
    """Runs a Pregel test in a separate process and stores the result in return_dict."""
    time_taken, load = measure_pregel_performance(pregel_class, vertices_data, num_workers)
    return_dict[key] = (time_taken, load)

if __name__ == "__main__":  

    # run exps for different worker sizes (keeping nodes fixed)
    fixed_num_nodes = 10000
    vertices_fixed = [(j, 1.0 / fixed_num_nodes) for j in range(fixed_num_nodes)]  # Convert to tuples

    for i, num_workers in enumerate(NUM_WORKERS_LIST):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        process2 = multiprocessing.Process(target=run_pregel_test, args=(BalancedPregel, vertices_fixed, num_workers, return_dict, "balanced"))
        process1 = multiprocessing.Process(target=run_pregel_test, args=(Pregel, vertices_fixed, num_workers, return_dict, "original"))

        process1.start()
        process2.start()

        process1.join()
        process2.join()

        balanced_time, balanced_load = return_dict["balanced"]
        original_time, original_load = return_dict["original"]

        # if i > 0:
        execution_time_workers["original"].append(original_time)
        execution_time_workers["balanced"].append(balanced_time)
        speedup_workers["original"].append(1.0)
        speedup_workers["balanced"].append(original_time / balanced_time if balanced_time > 0 else float('inf'))
        load_balancing_workers["original"].append(original_load)
        load_balancing_workers["balanced"].append(balanced_load)

        gc.collect()  

    # run exp for different graph sizes (keeping workers fixed)
    fixed_num_workers = 6

    for i, num_nodes in enumerate(NUM_NODES_LIST):
        vertices_varied = [(j, 1.0 / num_nodes) for j in range(num_nodes)]  # Convert to tuples

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        process2 = multiprocessing.Process(target=run_pregel_test, args=(BalancedPregel, vertices_varied, fixed_num_workers, return_dict, "balanced"))
        process1 = multiprocessing.Process(target=run_pregel_test, args=(Pregel, vertices_varied, fixed_num_workers, return_dict, "original"))

        process1.start()
        process2.start()

        process1.join()
        process2.join()

        balanced_time, balanced_load = return_dict["balanced"]
        original_time, original_load = return_dict["original"]

        # if i > 0:
        execution_time_nodes["original"].append(original_time)
        execution_time_nodes["balanced"].append(balanced_time)
        speedup_nodes["original"].append(1.0)
        speedup_nodes["balanced"].append(original_time / balanced_time if balanced_time > 0 else float('inf'))
        load_balancing_nodes["original"].append(original_load)
        load_balancing_nodes["balanced"].append(balanced_load)

        gc.collect() 

    import os
    import datetime
    os.mkdir("results") if not os.path.exists("results") else None
    current_datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")
    csv_filename = f"results/pregel_experiment_results_{current_datetime}.csv"

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment Type", "Num Workers", "Num Nodes", "Original Time", "Balanced Time", 
                         "Original Speedup", "Balanced Speedup", "Original Load", "Balanced Load"])

        for i in range(len(NUM_WORKERS_LIST)):  # -1 because we skipped first result
            writer.writerow(["Workers", NUM_WORKERS_LIST[i], fixed_num_nodes, 
                             execution_time_workers["original"][i], execution_time_workers["balanced"][i], 
                             speedup_workers["original"][i], speedup_workers["balanced"][i], 
                             load_balancing_workers["original"][i], load_balancing_workers["balanced"][i]])

        for i in range(len(NUM_NODES_LIST)):  # -1 because we skipped first result
            writer.writerow(["Nodes", fixed_num_workers, NUM_NODES_LIST[i], 
                             execution_time_nodes["original"][i], execution_time_nodes["balanced"][i], 
                             speedup_nodes["original"][i], speedup_nodes["balanced"][i], 
                             load_balancing_nodes["original"][i], load_balancing_nodes["balanced"][i]])

    print(f"Experiment results saved to {csv_filename}")

    import subprocess
    try:
        subprocess.run(["python", "genplot.py", csv_filename], check=True)
        print("Plotting script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the plotting script: {e}")