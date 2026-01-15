"""
BMCA - Best Master Clock Algorithm (IEEE 802.1AS/1588)
Tuna Girisken, Ege University
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import IntEnum


class ClockClass(IntEnum):
    """IEEE 1588/802.1AS Clock Classes"""
    GPS = 6              # Primary reference (GPS, GNSS)
    ARB = 7              # Arbitrary time scale
    AT = 13              # Application-specific time scale
    DFT = 52             # Default (local oscillator)
    PRC = 187            # Primary reference clock (telecom)
    DEFAULT = 248        # Default (worst quality)
    SLAVE_ONLY = 255     # Slave-only clock


class ClockAccuracy(IntEnum):
    """IEEE 1588 Clock Accuracy Levels"""
    ACC_25NS = 0x20      # Accurate to 25ns
    ACC_100NS = 0x21     # Accurate to 100ns
    ACC_250NS = 0x22     # Accurate to 250ns
    ACC_1US = 0x23       # Accurate to 1μs
    ACC_10US = 0x25      # Accurate to 10μs
    ACC_100US = 0x27     # Accurate to 100μs
    ACC_1MS = 0x29       # Accurate to 1ms
    ACC_10MS = 0x2B      # Accurate to 10ms
    ACC_100MS = 0x2D     # Accurate to 100ms
    ACC_UNKNOWN = 0x31   # Unknown accuracy (worst)


@dataclass
class ClockQuality:
    """IEEE 802.1AS Clock Quality Structure"""
    clock_class: int = ClockClass.DEFAULT
    clock_accuracy: int = ClockAccuracy.ACC_UNKNOWN
    offset_scaled_log_variance: int = 0xFFFF  # Allan variance, lower is better
    
    def __str__(self):
        return f"ClockQuality(class={self.clock_class}, acc={self.clock_accuracy}, var={self.offset_scaled_log_variance})"


@dataclass
class BMCADataSet:
    """Complete BMCA Dataset for Clock Comparison"""
    node_id: int
    priority1: int = 255                       # Priority 1 (0-255, lower is better)
    clock_quality: ClockQuality = field(default_factory=ClockQuality)
    priority2: int = 255                       # Priority 2 (tiebreaker)
    
    def __str__(self):
        return (f"Node {self.node_id} [P1={self.priority1}, "
                f"Class={self.clock_quality.clock_class}, "
                f"Acc={self.clock_quality.clock_accuracy}, "
                f"Var={self.clock_quality.offset_scaled_log_variance}, "
                f"P2={self.priority2}]")


class StandardBMCA:
    """
    Standard BMCA Implementation (IEEE 802.1AS-2020)
    
    Best Master Clock Algorithm - Distributed leader election that selects
    Grand Master based on clock quality metrics:
    1. Priority1 (lower is better)
    2. Clock Class (lower is better: 6=GPS, 248=DEFAULT)
    3. Clock Accuracy (lower is better)
    4. Offset Scaled Log Variance (lower is better)
    5. Priority2 (lower is better)
    6. Node ID (lower is better, tiebreaker)
    
    Algorithm: Flooding-based distributed consensus
    - Each node broadcasts its best known GM candidate
    - Nodes compare received candidates with local best
    - Converges in D rounds where D = network diameter
    """
    
    def __init__(self, graph, node_properties: Optional[Dict[int, BMCADataSet]] = None):
        """Initialize BMCA with network graph and optional clock properties."""
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.diameter = self._compute_diameter()
        
        # Initialize clock properties for each node
        self.local_clocks = self._initialize_clocks(node_properties)
        
        # Each node's current best GM candidate
        self.best_master: Dict[int, BMCADataSet] = {}
        for node_id in self.graph.nodes():
            self.best_master[node_id] = self.local_clocks[node_id]
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.rounds_completed = 0
        self.elected_gm: Optional[int] = None
    
    def _compute_diameter(self) -> int:
        """Compute network diameter (longest shortest path)"""
        import networkx as nx
        try:
            if nx.is_connected(self.graph):
                return nx.diameter(self.graph)
            else:
                # For disconnected graphs, use max diameter of components
                components = nx.connected_components(self.graph)
                max_diam = 0
                for comp in components:
                    subgraph = self.graph.subgraph(comp)
                    if len(comp) > 1:
                        max_diam = max(max_diam, nx.diameter(subgraph))
                return max_diam if max_diam > 0 else self.num_nodes
        except:
            # Fallback to number of nodes
            return self.num_nodes
    
    def _initialize_clocks(self, node_properties: Optional[Dict[int, BMCADataSet]]) -> Dict[int, BMCADataSet]:
        """Initialize clock properties for all nodes"""
        if node_properties:
            return node_properties.copy()
        
        # Generate default properties
        clocks = {}
        for node_id in self.graph.nodes():
            clocks[node_id] = BMCADataSet(
                node_id=node_id,
                priority1=255,  # Default: lowest priority
                clock_quality=ClockQuality(),
                priority2=255
            )
        return clocks
    
    def compare_bmca(self, a: BMCADataSet, b: BMCADataSet) -> bool:
        """
        IEEE 802.1AS BMCA Comparison
        
        Returns True if 'a' is better than 'b'
        
        Comparison order:
        1. Priority1 (lower is better)
        2. Clock Class (lower is better)
        3. Clock Accuracy (lower is better)
        4. Offset Scaled Log Variance (lower is better)
        5. Priority2 (lower is better)
        6. Node ID (lower is better, tiebreaker)
        """
        # 1. Priority1
        if a.priority1 != b.priority1:
            return a.priority1 < b.priority1
        
        # 2. Clock Class
        if a.clock_quality.clock_class != b.clock_quality.clock_class:
            return a.clock_quality.clock_class < b.clock_quality.clock_class
        
        # 3. Clock Accuracy
        if a.clock_quality.clock_accuracy != b.clock_quality.clock_accuracy:
            return a.clock_quality.clock_accuracy < b.clock_quality.clock_accuracy
        
        # 4. Offset Scaled Log Variance
        if a.clock_quality.offset_scaled_log_variance != b.clock_quality.offset_scaled_log_variance:
            return a.clock_quality.offset_scaled_log_variance < b.clock_quality.offset_scaled_log_variance
        
        # 5. Priority2
        if a.priority2 != b.priority2:
            return a.priority2 < b.priority2
        
        # 6. Node ID (tiebreaker)
        return a.node_id < b.node_id
    
    def run_round(self) -> bool:
        """
        Execute one round of BMCA
        
        Returns True if any node updated its best master
        """
        updated = False
        new_best_master = {}
        
        # Each node processes messages from neighbors
        for node_id in self.graph.nodes():
            current_best = self.best_master[node_id]
            
            # Receive announces from all neighbors
            for neighbor_id in self.graph.neighbors(node_id):
                neighbor_best = self.best_master[neighbor_id]
                self.messages_received += 1
                
                # Compare with current best
                if self.compare_bmca(neighbor_best, current_best):
                    current_best = neighbor_best
                    updated = True
            
            new_best_master[node_id] = current_best
            
            # Each node broadcasts to neighbors (count messages)
            self.messages_sent += len(list(self.graph.neighbors(node_id)))
        
        # Update best masters for next round
        self.best_master = new_best_master
        self.rounds_completed += 1
        
        return updated
    
    def run(self, max_rounds: Optional[int] = None, verbose: bool = False) -> int:
        """Run BMCA until convergence and return elected GM node ID."""
        if max_rounds is None:
            max_rounds = self.diameter + 1
        
        if verbose:
            print(f"Starting BMCA on graph with {self.num_nodes} nodes, diameter={self.diameter}")
            print(f"Max rounds: {max_rounds}")
        
        for round_num in range(max_rounds):
            updated = self.run_round()
            
            if verbose:
                print(f"Round {round_num + 1}/{max_rounds}: {'Updated' if updated else 'Converged'}")
            
            # Early termination if converged
            if not updated and round_num > 0:
                if verbose:
                    print(f"Converged early at round {round_num + 1}")
                break
        
        # Verify convergence - all nodes should agree on GM
        gm_votes = {}
        for node_id in self.graph.nodes():
            gm_id = self.best_master[node_id].node_id
            gm_votes[gm_id] = gm_votes.get(gm_id, 0) + 1
        
        if len(gm_votes) > 1:
            print(f"WARNING: BMCA did not converge! Multiple GMs elected: {gm_votes}")
            # Return the most voted GM
            self.elected_gm = max(gm_votes.items(), key=lambda x: x[1])[0]
        else:
            self.elected_gm = list(gm_votes.keys())[0]
        
        if verbose:
            print(f"\nElected Grand Master: Node {self.elected_gm}")
            print(f"Total messages: {self.messages_sent}")
            print(f"Rounds completed: {self.rounds_completed}")
        
        return self.elected_gm
    
    def get_stats(self) -> Dict:
        """Return algorithm statistics"""
        return {
            'elected_gm': self.elected_gm,
            'rounds': self.rounds_completed,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'diameter': self.diameter,
            'num_nodes': self.num_nodes
        }
