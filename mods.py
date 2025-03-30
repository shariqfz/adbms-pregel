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
