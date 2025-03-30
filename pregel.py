# original

"""pregel.py - A Python 3.10+ implementation of a toy Pregel system for graph processing."""

# from __future__ import annotations
from typing import List, Dict, Tuple, DefaultDict, Any
import collections
import threading
from dataclasses import dataclass

import random
import numpy as np
seed=42
random.seed = seed
np.random.seed = seed

@dataclass
class Vertex:
    id: int
    value: float
    out_vertices: List['Vertex']
    incoming_messages: List[Tuple['Vertex', float]] = None
    outgoing_messages: List[Tuple['Vertex', float]] = None
    active: bool = True
    superstep: int = 0

    def __post_init__(self):
        self.incoming_messages = []
        self.outgoing_messages = []

    def __hash__(self):
        return hash(self.id)

class Pregel:
    def __init__(self, vertices: List[Vertex], num_workers: int):
        self.vertices = vertices
        self.num_workers = num_workers
        self.partition: DefaultDict[int, List[Vertex]] = collections.defaultdict(list)
        self.total_messages = 0  
        self.supersteps = 0  

    def run(self) -> None:
        """Runs the Pregel computation until completion."""
        self.partition = self.partition_vertices()
        while self.check_active():
            self.superstep()
            self.redistribute_messages()
            self.supersteps += 1  

        return self.supersteps, self.total_messages  


    def partition_vertices(self) -> DefaultDict[int, List[Vertex]]:
        """Partitions vertices across workers using consistent hashing."""
        partition = collections.defaultdict(list)
        for vertex in self.vertices:
            partition[self.worker(vertex)].append(vertex)
        return partition

    def worker(self, vertex: Vertex) -> int:
        """Determines which worker handles this vertex."""
        return hash(vertex) % self.num_workers

    def superstep(self) -> None:
        """Executes one superstep using worker threads."""
        workers = []
        for vertex_list in self.partition.values():
            worker = Worker(vertex_list)
            workers.append(worker)
            worker.start()
        
        for worker in workers:
            worker.join()

    def redistribute_messages(self) -> None:
        """Routes messages between vertices after each superstep."""
        for vertex in self.vertices:
            vertex.superstep += 1
            vertex.incoming_messages.clear()
        
        for vertex in self.vertices:
            self.total_messages += len(vertex.outgoing_messages)
            for (receiver, message) in vertex.outgoing_messages:
                receiver.incoming_messages.append((vertex, message))
            vertex.outgoing_messages.clear()

    def check_active(self) -> bool:
        """Checks if any vertices are still active."""
        return any(vertex.active for vertex in self.vertices)

class Worker(threading.Thread):
    def __init__(self, vertices: List[Vertex]):
        super().__init__()
        self.vertices = vertices

    def run(self) -> None:
        """Executes the superstep for assigned vertices."""
        for vertex in self.vertices:
            if vertex.active:
                vertex.update()



# original

"""pagerank.py - PageRank implementation using the Pregel framework."""

from typing import List
import random
import numpy as np
from numpy.linalg import norm
# from pregel import Vertex, Pregel

seed=42
random.seed = seed
np.random.seed = seed
# Configuration
NUM_WORKERS = 4
NUM_VERTICES = 10
DAMPING_FACTOR = 0.85
MAX_SUPERSTEPS = 50

def pregel() -> None:
    """Main execution comparing Pregel vs matrix PageRank."""
    vertices = [PageRankVertex(j, 1.0/NUM_VERTICES, []) 
               for j in range(NUM_VERTICES)]
    create_edges(vertices)
    
    pr_test = pagerank_test(vertices)
    print(f"Test computation of pagerank:\n{pr_test}")
    
    pr_pregel = pagerank_pregel(vertices)
    print(f"Pregel computation of pagerank:\n{pr_pregel}")
    
    diff = pr_pregel - pr_test
    print(f"Difference between the two pagerank vectors:\n{diff}")
    print(f"The norm of the difference is: {norm(diff):.2e}")

# def create_edges(vertices: List[Vertex]) -> None:
#     """Creates random edges between vertices."""
#     min_size = 4
#     max_size = len(vertices) // 10
#     if max_size < min_size:
#         max_size = min_size

#     for vertex in vertices:
#         # vertex.out_vertices = random.sample(vertices, min(4, len(vertices)))
#         sample_size = random.randint(min_size, max_size)
#         vertex.out_vertices = random.sample(vertices, sample_size)
def create_edges(vertices: List[Vertex], exponent=2.5) -> None:
    """
    Creates edges following a power-law degree distribution.
    
    - Few nodes will have a **high number of edges** (hubs).
    - Most nodes will have a **low number of edges**.
    - `exponent` controls the skew (higher = fewer hubs, lower = more hubs).
    """
    num_nodes = len(vertices)

    # Generate degrees from a power-law distribution
    degrees = np.random.zipf(exponent, num_nodes)  # Zipf distribution for power-law
    max_degree = num_nodes // 5  # Prevent extreme values
    degrees = np.clip(degrees, 1, max_degree)  # Limit max degree

    # Shuffle nodes so high-degree nodes are randomly placed
    random.shuffle(vertices)

    for i, vertex in enumerate(vertices):
        num_edges = degrees[i]
        num_edges = min(num_edges, num_nodes - 1)  # Avoid selecting all nodes

        # Select neighbors without self-loops
        vertex.out_vertices = random.sample([v for v in vertices if v != vertex], num_edges)


def pagerank_test(vertices: List[Vertex]) -> np.ndarray:
    """Computes PageRank using matrix operations."""
    I = np.eye(NUM_VERTICES)
    G = np.zeros((NUM_VERTICES, NUM_VERTICES))
    
    for vertex in vertices:
        num_out = len(vertex.out_vertices)
        for neighbor in vertex.out_vertices:
            G[neighbor.id, vertex.id] = 1.0 / num_out
    
    P = (1.0/NUM_VERTICES) * np.ones((NUM_VERTICES, 1))
    return 0.15 * np.linalg.inv(I - DAMPING_FACTOR*G) @ P

def pagerank_pregel(vertices: List[Vertex]) -> np.ndarray:
    """Computes PageRank using the Pregel framework."""
    p = Pregel(vertices, NUM_WORKERS)
    p.run()
    return np.array([vertex.value for vertex in p.vertices]).reshape(-1, 1)

class PageRankVertex(Vertex):
    def update(self) -> None:
        """Vertex update rule for PageRank."""
        if self.superstep < MAX_SUPERSTEPS:
            incoming_sum = sum(msg for (_, msg) in self.incoming_messages)
            self.value = (0.15 / NUM_VERTICES + 
                         DAMPING_FACTOR * incoming_sum)
            
            outgoing_value = self.value / len(self.out_vertices)
            self.outgoing_messages = [
                (neighbor, outgoing_value) 
                for neighbor in self.out_vertices
            ]
        else:
            self.active = False

# pregel()