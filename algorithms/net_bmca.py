"""
NetBMCA - Network-Aware Best Master Clock Algorithm
Tuna Girisken, Ege University
"""

from typing import Dict, Optional
import networkx as nx
from .standard_bmca import StandardBMCA, BMCADataSet
import numpy as np


class CentralityMetrics:
    """Network centrality metrics for topology-aware GM selection"""
    def __init__(self):
        self.betweenness: float = 0.0
        self.closeness: float = 0.0
        self.eigenvector: float = 0.0
        self.degree: float = 0.0
    
    def __str__(self):
        return (f"CentralityMetrics(betweenness={self.betweenness:.4f}, "
                f"closeness={self.closeness:.4f}, eigenvector={self.eigenvector:.4f})")


class NetBMCA(StandardBMCA):
    """
    Network-Aware Best Master Clock Algorithm
    
    Extends IEEE 802.1AS BMCA by incorporating centrality metrics
    (betweenness, closeness, eigenvector) to select topologically
    optimal grandmasters for minimizing synchronization path length.
    """
    
    def __init__(self, 
                 graph, 
                 node_properties: Optional[Dict[int, BMCADataSet]] = None,
                 enable_adaptive_weights: bool = True,
                 w_priority: float = 0.30,
                 w_class: float = 0.25,
                 w_betweenness: float = 0.25,
                 w_closeness: float = 0.15,
                 w_eigenvector: float = 0.05):
        """Initialize NetBMCA with network graph and weight configuration."""
        super().__init__(graph, node_properties)
        
        # Network characteristics
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.density = nx.density(graph)
        
        # Initialize weights
        if enable_adaptive_weights:
            self._compute_adaptive_weights()
        else:
            self.w_priority = w_priority
            self.w_class = w_class
            self.w_betweenness = w_betweenness
            self.w_closeness = w_closeness
            self.w_eigenvector = w_eigenvector
        
        # Normalize weights to sum to 1.0
        self._normalize_weights()
        
        # Compute centrality metrics
        self.node_metrics: Dict[int, CentralityMetrics] = {}
        self._compute_centrality_metrics()
    
    def _compute_adaptive_weights(self):
        """Tune weights based on network size and density."""
        n = self.num_nodes
        d = self.density
        
        if n < 20:
            # Small network: clock quality matters more
            self.w_priority = 0.35
            self.w_class = 0.30
            self.w_betweenness = 0.20
            self.w_closeness = 0.10
            self.w_eigenvector = 0.05
        elif n < 100:
            # Medium network: balanced approach
            self.w_priority = 0.30
            self.w_class = 0.25
            self.w_betweenness = 0.25
            self.w_closeness = 0.15
            self.w_eigenvector = 0.05
        else:
            # Large network: topology matters more
            self.w_priority = 0.25
            self.w_class = 0.20
            self.w_betweenness = 0.30
            self.w_closeness = 0.20
            self.w_eigenvector = 0.05
        
        # Adjust for density
        if d > 0.3:  # Dense network
            # Betweenness less discriminative in dense graphs
            self.w_betweenness *= 0.7
            self.w_closeness *= 1.3
        elif d < 0.05:  # Sparse network
            # Betweenness more important
            self.w_betweenness *= 1.3
            self.w_closeness *= 0.7
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = (self.w_priority + self.w_class + self.w_betweenness + 
                self.w_closeness + self.w_eigenvector)
        
        if total > 0:
            self.w_priority /= total
            self.w_class /= total
            self.w_betweenness /= total
            self.w_closeness /= total
            self.w_eigenvector /= total
    
    def _compute_centrality_metrics(self):
        """Compute betweenness, closeness, and eigenvector centrality for all nodes."""
        print(f"Computing centrality metrics for {self.num_nodes} nodes...")
        
        # Betweenness centrality (normalized)
        betweenness = nx.betweenness_centrality(self.graph, normalized=True)
        
        # Closeness centrality (handle disconnected graphs)
        if nx.is_connected(self.graph):
            closeness = nx.closeness_centrality(self.graph)
        else:
            # Compute per component and normalize
            closeness = {}
            for component in nx.connected_components(self.graph):
                subgraph = self.graph.subgraph(component)
                comp_closeness = nx.closeness_centrality(subgraph)
                closeness.update(comp_closeness)
        
        # Degree centrality (normalized)
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Eigenvector centrality with fallback
        try:
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            # Fallback: use PageRank as proxy
            try:
                eigenvector = nx.pagerank(self.graph)
            except:
                # Last resort: use degree as proxy
                eigenvector = degree_centrality
        
        # Store metrics for all nodes
        for node_id in self.graph.nodes():
            metrics = CentralityMetrics()
            metrics.betweenness = betweenness.get(node_id, 0.0)
            metrics.closeness = closeness.get(node_id, 0.0)
            metrics.eigenvector = eigenvector.get(node_id, 0.0)
            metrics.degree = degree_centrality.get(node_id, 0.0)
            self.node_metrics[node_id] = metrics
        
        # Print statistics
        avg_betweenness = np.mean([m.betweenness for m in self.node_metrics.values()])
        avg_closeness = np.mean([m.closeness for m in self.node_metrics.values()])
        print(f"  Average betweenness: {avg_betweenness:.4f}")
        print(f"  Average closeness: {avg_closeness:.4f}")
    
    def get_centrality(self, node_id: int) -> CentralityMetrics:
        """Get centrality metrics for a specific node"""
        return self.node_metrics.get(node_id, CentralityMetrics())
    
    def compute_node_score(self, node: BMCADataSet) -> float:
        """Calculate weighted score combining clock quality and centrality metrics."""
        metrics = self.get_centrality(node.node_id)
        
        # Clock quality scores (normalize and invert: lower is better)
        priority_score = 1.0 - (node.priority1 / 255.0)
        class_score = 1.0 - (min(node.clock_quality.clock_class, 255) / 255.0)
        
        # Centrality scores (higher is better)
        betweenness_score = metrics.betweenness
        closeness_score = metrics.closeness
        eigenvector_score = metrics.eigenvector
        
        # Weighted sum
        total_score = (self.w_priority * priority_score +
                      self.w_class * class_score +
                      self.w_betweenness * betweenness_score +
                      self.w_closeness * closeness_score +
                      self.w_eigenvector * eigenvector_score)
        
        return total_score
    
    def compare_bmca(self, a: BMCADataSet, b: BMCADataSet) -> bool:
        """Compare two nodes using weighted scoring. Returns True if a is better than b."""
        score_a = self.compute_node_score(a)
        score_b = self.compute_node_score(b)
        
        # Higher score wins
        if abs(score_a - score_b) > 1e-6:  # Epsilon for floating point comparison
            return score_a > score_b
        
        # Tie-breaker: use node ID (lower wins, ensures determinism)
        return a.node_id < b.node_id
    
    def run(self, max_rounds: Optional[int] = None, verbose: bool = False) -> int:
        """Execute NetBMCA algorithm and return elected GM node ID."""
        if verbose:
            print("\n" + "="*60)
            print("CA-BMCA: Centrality-Aware Best Master Clock Algorithm")
            print("="*60)
            print(f"Network: {self.num_nodes} nodes, {self.num_edges} edges")
            print(f"Density: {self.density:.4f}, Diameter: {self.diameter}")
            print(f"\nWeights: P1={self.w_priority:.3f}, Class={self.w_class:.3f}, "
                  f"Bet={self.w_betweenness:.3f}, Close={self.w_closeness:.3f}, "
                  f"Eigen={self.w_eigenvector:.3f}")
            
            # Show top 5 candidates
            print("\nTop 5 GM Candidates by Score:")
            scores = [(nid, self.compute_node_score(self.local_clocks[nid])) 
                     for nid in self.graph.nodes()]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (node_id, score) in enumerate(scores[:5], 1):
                metrics = self.get_centrality(node_id)
                clock = self.local_clocks[node_id]
                print(f"  {rank}. Node {node_id:3d} | Score: {score:.4f} | "
                      f"P1={clock.priority1:3d} Class={clock.clock_quality.clock_class:3d} | "
                      f"Bet={metrics.betweenness:.3f} Close={metrics.closeness:.3f}")
        
        # Run BMCA with CA comparison function
        gm = super().run(max_rounds, verbose=verbose)
        
        # Print GM details
        if verbose and self.elected_gm is not None:
            gm_metrics = self.get_centrality(self.elected_gm)
            gm_score = self.compute_node_score(self.local_clocks[self.elected_gm])
            gm_clock = self.local_clocks[self.elected_gm]
            
            print("\n" + "="*60)
            print("ELECTED GRAND MASTER")
            print("="*60)
            print(f"Node ID: {self.elected_gm}")
            print(f"Overall Score: {gm_score:.4f}")
            print(f"\nClock Properties:")
            print(f"  Priority1: {gm_clock.priority1}")
            print(f"  Clock Class: {gm_clock.clock_quality.clock_class}")
            print(f"  Clock Accuracy: {gm_clock.clock_quality.clock_accuracy}")
            print(f"\nCentrality Metrics:")
            print(f"  Betweenness: {gm_metrics.betweenness:.4f}")
            print(f"  Closeness: {gm_metrics.closeness:.4f}")
            print(f"  Eigenvector: {gm_metrics.eigenvector:.4f}")
            print(f"  Degree: {gm_metrics.degree:.4f}")
            print("="*60)
        
        return gm
    
    def get_stats(self) -> Dict:
        """Return algorithm statistics including weights and centrality metrics."""
        stats = super().get_stats()
        
        # Add weight configuration
        stats['weights'] = {
            'priority': self.w_priority,
            'class': self.w_class,
            'betweenness': self.w_betweenness,
            'closeness': self.w_closeness,
            'eigenvector': self.w_eigenvector
        }
        
        # Add GM centrality metrics
        if self.elected_gm is not None:
            gm_metrics = self.get_centrality(self.elected_gm)
            stats['gm_centrality'] = {
                'betweenness': gm_metrics.betweenness,
                'closeness': gm_metrics.closeness,
                'eigenvector': gm_metrics.eigenvector,
                'degree': gm_metrics.degree,
                'score': self.compute_node_score(self.local_clocks[self.elected_gm])
            }
        
        # Network characteristics
        stats['network'] = {
            'nodes': self.num_nodes,
            'edges': self.num_edges,
            'density': self.density,
            'diameter': self.diameter
        }
        
        return stats
