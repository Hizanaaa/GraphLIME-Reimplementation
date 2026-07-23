"""
Neighborhood extraction utilities for GraphLIME.

This module extracts k-hop subgraphs centered on a target node.
"""

from __future__ import annotations

from torch import Tensor
from torch_geometric.utils import k_hop_subgraph


class NeighborhoodExtractor:
    """
    Extract k-hop neighborhoods from a graph.
    """

    def __init__(self, num_hops: int = 2) -> None:
        self.num_hops = num_hops

    def extract(
        self,
        node_idx: int,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Extract the k-hop neighborhood.

        Parameters
        ----------
        node_idx : int
            Target node index.

        edge_index : Tensor
            Graph connectivity.

        Returns
        -------
        subset : Tensor
            Node indices in the neighborhood.

        sub_edge_index : Tensor
            Relabeled edge index for the subgraph.

        mapping : Tensor
            Index of the target node within the subgraph.
        """

        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=self.num_hops,
            edge_index=edge_index,
            relabel_nodes=True,
        )

        return subset, sub_edge_index, mapping
    
if __name__ == "__main__":

    from datasets.loader import get_dataset

    dataset, data = get_dataset("cora")

    extractor = NeighborhoodExtractor(num_hops=2)

    subset, edge_index, mapping = extractor.extract(
        node_idx=100,
        edge_index=data.edge_index,
    )

    print("Target Node:", 100)
    print("Neighborhood Size:", len(subset))
    print("Subgraph Nodes:", subset.tolist())
    print("Subgraph Edge Shape:", edge_index.shape)

    assert mapping.item() == 0

    print("\nAll tests passed.")
