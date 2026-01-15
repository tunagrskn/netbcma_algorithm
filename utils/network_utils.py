"""
Network utilities for BMCA analysis
Tuna Girisken, Ege University
"""

import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def calculate_ptp_round_trip_delay(G, gm):
    """
    Calculate PTP round-trip delay (forward + backward path) for each node.
    Returns dict mapping node_id to delay in hops.
    """
    round_trip_delays = {}
    
    for node in G.nodes():
        if node == gm:
            round_trip_delays[node] = 0
            continue
        
        # Forward path: GM → Slave (Sync + Follow_Up)
        try:
            forward_path = nx.shortest_path(G, gm, node)
            forward_hops = len(forward_path) - 1
        except nx.NetworkXNoPath:
            forward_hops = float('inf')
        
        # Backward path: Slave → GM (Delay_Req)
        try:
            backward_path = nx.shortest_path(G, node, gm)
            backward_hops = len(backward_path) - 1
        except nx.NetworkXNoPath:
            backward_hops = float('inf')
        
        # Round-trip delay (in hops)
        round_trip_delays[node] = forward_hops + backward_hops
    
    return round_trip_delays


def load_network(filepath: str) -> nx.Graph:
    """Load network from GML file."""
    G = nx.read_gml(filepath, label='id')
    # Ensure node IDs are integers
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G

def compute_centrality_metrics(G: nx.Graph, fast_mode: bool = False) -> Dict[int, Dict[str, float]]:
    """Compute centrality metrics for all nodes.
    
    Args:
        G: NetworkX graph
        fast_mode: Use degree as approximation for large graphs
        
    Returns:
        Dictionary mapping node_id -> {betweenness, closeness, degree}
    """
    n = len(G.nodes())
    metrics = {}
    
    if fast_mode or n > 100:
        # Use degree as approximation
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        for node in G.nodes():
            metrics[node] = {
                'betweenness': degrees[node] / max_degree,
                'closeness': degrees[node] / max_degree,
                'degree': degrees[node] / max_degree
            }
    else:
        # Compute exact centrality
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        for node in G.nodes():
            metrics[node] = {
                'betweenness': betweenness.get(node, 0),
                'closeness': closeness.get(node, 0),
                'degree': degrees[node] / max_degree
            }
    
    return metrics

def compute_hop_distances(G: nx.Graph, source: int) -> Dict[int, int]:
    """Compute hop distances from source to all nodes.
    
    Args:
        G: NetworkX graph
        source: Source node
        
    Returns:
        Dictionary mapping node_id -> hop_distance
    """
    if source not in G.nodes():
        return {node: 0 for node in G.nodes()}
    
    if nx.is_connected(G):
        return nx.single_source_shortest_path_length(G, source)
    else:
        # Handle disconnected graphs
        component = nx.node_connected_component(G, source)
        subgraph = G.subgraph(component)
        distances = nx.single_source_shortest_path_length(subgraph, source)
        # Nodes in other components get infinite distance (represented as max)
        for node in G.nodes():
            if node not in distances:
                distances[node] = float('inf')
        return distances

def get_network_statistics(G: nx.Graph) -> Dict[str, any]:
    """Get basic network statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G)
    }
    
    if nx.is_connected(G):
        stats['diameter'] = nx.diameter(G)
        stats['avg_clustering'] = nx.average_clustering(G)
        stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
    else:
        stats['diameter'] = None
        stats['avg_clustering'] = nx.average_clustering(G)
        stats['avg_shortest_path'] = None
        stats['num_components'] = nx.number_connected_components(G)
    
    degrees = [d for n, d in G.degree()]
    stats['avg_degree'] = sum(degrees) / len(degrees) if degrees else 0
    stats['max_degree'] = max(degrees) if degrees else 0
    stats['min_degree'] = min(degrees) if degrees else 0
    
    return stats

def find_networks(directory: str, pattern: str = "*.gml") -> List[Path]:
    """Find all network files in directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern (default: *.gml)
        
    Returns:
        List of Path objects
    """
    path = Path(directory)
    return sorted(path.glob(pattern))

__all__ = [
    'load_network',
    'compute_centrality_metrics',
    'compute_hop_distances',
    'get_network_statistics',
    'find_networks'
]
