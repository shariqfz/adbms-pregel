import collections
from typing import List, DefaultDict
from pregel import Pregel, Vertex


# degree partioning
class BalancedPregel(Pregel):
    def partition_vertices(self) -> DefaultDict[int, List[Vertex]]:
        """Partitions vertices across workers using degree-based load balancing."""
        partition = collections.defaultdict(list)
        
        # sort vertices by degree (high-degree first)
        sorted_vertices = sorted(self.vertices, key=lambda v: len(v.out_vertices), reverse=True)
        
        # distribute vertices among workers in a round-robin manner
        for i, vertex in enumerate(sorted_vertices):
            worker_id = i % self.num_workers  
            partition[worker_id].append(vertex)
        
        return partition
    
# message aggregation
class AggregatedMessagePregel(Pregel):
    def run(self):
        supersteps = 0
        active_vertices = set(self.vertices)
        total_messages = 0

        while active_vertices:
            new_active = set()
            message_count = 0

            for worker_vertices in self.partition.values():
                aggregated_messages = {}  # Dictionary to store aggregated messages
                
                for vertex in worker_vertices:
                    if vertex.active:
                        vertex.compute()
                        
                        # Aggregate messages before sending
                        for neighbor in vertex.out_vertices:
                            if neighbor.id not in aggregated_messages:
                                aggregated_messages[neighbor.id] = 0
                            aggregated_messages[neighbor.id] += vertex.value

                # Send only one aggregated message per recipient
                for neighbor_id, aggregated_value in aggregated_messages.items():
                    message_count += 1  # One message per unique neighbor

                new_active.update(worker_vertices)
                
            total_messages += message_count
            active_vertices = new_active
            supersteps += 1

        return supersteps, total_messages


# edge-cut partitioning
import networkx as nx
class PartitionedPregel(Pregel):
    def _partition_graph(self):
        """Use METIS-style edge-cut partitioning to minimize cross-worker communication."""
        num_vertices = len(self.vertices)
        graph = nx.Graph()

        # Build NetworkX graph for partitioning
        for vertex in self.vertices:
            graph.add_node(vertex.id)
            for neighbor in vertex.out_vertices:
                graph.add_edge(vertex.id, neighbor.id)

        # Use METIS or a heuristic partitioning
        partition_labels = nx.algorithms.community.kernighan_lin_bisection(graph)  # Approximate METIS

        partitions = {i: [] for i in range(self.num_workers)}
        for i, vertex in enumerate(self.vertices):
            partitions[partition_labels[i] % self.num_workers].append(vertex)

        return partitions

    def run(self):
        supersteps = 0
        active_vertices = set(self.vertices)
        total_messages = 0

        while active_vertices:
            new_active = set()
            message_count = 0

            for worker_id, worker_vertices in self.partition.items():
                local_messages = 0
                for vertex in worker_vertices:
                    if vertex.active:
                        vertex.compute()
                        
                        for neighbor in vertex.out_vertices:
                            if neighbor in worker_vertices:
                                local_messages += 1  # Local message
                            else:
                                message_count += 1  # Inter-worker message

                new_active.update(worker_vertices)

            total_messages += message_count
            active_vertices = new_active
            supersteps += 1

        return supersteps, total_messages

