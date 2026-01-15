"""
Synthetic TSN Network Generator
Tuna Girisken, Ege University
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple


class TSNTopologyGenerator:
    """Generates synthetic TSN network topologies"""
    
    @staticmethod
    def create_automotive_network(num_ecus: int = 20) -> Tuple[nx.Graph, str]:
        \"\"\"Create automotive Ethernet topology with gateway, domain controllers, and ECUs.\"\"\"
        G = nx.Graph()
        G.name = "Automotive_TSN"
        
        # Central Gateway
        gateway = 0
        G.add_node(gateway, label="Gateway", type="gateway")
        
        # Domain Controllers
        num_domains = min(4, max(3, num_ecus // 5))
        domain_controllers = list(range(1, num_domains + 1))
        
        for dc in domain_controllers:
            G.add_node(dc, label=f"Domain_Controller_{dc}", type="domain_controller")
            G.add_edge(gateway, dc)
        
        # ECUs per domain
        node_id = num_domains + 1
        ecus_per_domain = (num_ecus - num_domains - 1) // num_domains
        
        for dc in domain_controllers:
            # Star topology within domain
            for i in range(ecus_per_domain):
                ecu_id = node_id
                G.add_node(ecu_id, label=f"ECU_{ecu_id}", type="ecu", domain=dc)
                G.add_edge(dc, ecu_id)
                node_id += 1
        
        # Add some cross-domain connections (redundancy)
        if num_ecus > 15:
            # Connect some domain controllers to each other
            for i in range(len(domain_controllers) - 1):
                G.add_edge(domain_controllers[i], domain_controllers[i+1])
        
        description = f"Automotive TSN Network: {G.number_of_nodes()} nodes (1 gateway, {num_domains} domains, {num_ecus-num_domains-1} ECUs)"
        return G, description
    
    @staticmethod
    def create_industrial_network(num_devices: int = 30, ring: bool = True) -> Tuple[nx.Graph, str]:
        """
        Industrial Ethernet Topology (PROFINET, EtherCAT)
        
        Structure:
        - PLC/Master (1 node)
        - IO Devices (ring or line topology)
        - Redundant ring for reliability
        """
        G = nx.Graph()
        G.name = "Industrial_TSN"
        
        # PLC Master
        plc = 0
        G.add_node(plc, label="PLC_Master", type="master")
        
        # IO Devices
        devices = list(range(1, num_devices))
        
        if ring:
            # Ring topology (typical for PROFINET)
            for i in range(len(devices)):
                dev = devices[i]
                G.add_node(dev, label=f"IO_Device_{dev}", type="io_device")
                
                # Connect to previous device
                if i == 0:
                    G.add_edge(plc, dev)
                else:
                    G.add_edge(devices[i-1], dev)
            
            # Close the ring
            G.add_edge(devices[-1], plc)
            
            # Add redundant connections (every 5th device)
            for i in range(0, len(devices), 5):
                if i + 2 < len(devices):
                    G.add_edge(devices[i], devices[i+2])
        else:
            # Line/Daisy-chain topology (typical for EtherCAT)
            for i, dev in enumerate(devices):
                G.add_node(dev, label=f"IO_Device_{dev}", type="io_device")
                
                if i == 0:
                    G.add_edge(plc, dev)
                else:
                    G.add_edge(devices[i-1], dev)
        
        topology_type = "Ring" if ring else "Line"
        description = f"Industrial TSN Network ({topology_type}): {G.number_of_nodes()} nodes (1 PLC, {num_devices-1} IO devices)"
        return G, description
    
    @staticmethod
    def create_avionics_network(num_end_systems: int = 25) -> Tuple[nx.Graph, str]:
        """
        Avionics Ethernet Topology (AFDX - Avionics Full-Duplex Switched Ethernet)
        
        Structure:
        - Redundant switches (2 separate networks)
        - End Systems connected to both networks
        - Hierarchical tree structure
        """
        G = nx.Graph()
        G.name = "Avionics_TSN"
        
        # Two redundant root switches
        switch_a = 0
        switch_b = 1
        G.add_node(switch_a, label="Switch_A", type="switch", network="A")
        G.add_node(switch_b, label="Switch_B", type="switch", network="B")
        
        # Secondary switches
        num_secondary = max(2, num_end_systems // 8)
        secondary_switches_a = list(range(2, 2 + num_secondary))
        secondary_switches_b = list(range(2 + num_secondary, 2 + 2*num_secondary))
        
        for sw in secondary_switches_a:
            G.add_node(sw, label=f"Switch_A{sw}", type="switch", network="A")
            G.add_edge(switch_a, sw)
        
        for sw in secondary_switches_b:
            G.add_node(sw, label=f"Switch_B{sw}", type="switch", network="B")
            G.add_edge(switch_b, sw)
        
        # End systems (dual-homed to both networks)
        node_id = 2 + 2*num_secondary
        end_systems_per_switch = num_end_systems // num_secondary
        
        for i in range(num_secondary):
            sw_a = secondary_switches_a[i]
            sw_b = secondary_switches_b[i]
            
            for j in range(end_systems_per_switch):
                es_id = node_id
                G.add_node(es_id, label=f"End_System_{es_id}", type="end_system")
                # Dual redundancy
                G.add_edge(sw_a, es_id)
                G.add_edge(sw_b, es_id)
                node_id += 1
        
        description = f"Avionics TSN Network (AFDX): {G.number_of_nodes()} nodes (redundant topology, {num_end_systems} end systems)"
        return G, description
    
    @staticmethod
    def create_datacenter_network(num_servers: int = 40) -> Tuple[nx.Graph, str]:
        """
        Data Center Topology (Fat-Tree for TSN)
        
        Structure:
        - Core switches
        - Aggregation switches  
        - Top-of-Rack (ToR) switches
        - Servers
        """
        G = nx.Graph()
        G.name = "DataCenter_TSN"
        
        # Simple 3-tier architecture
        num_pods = max(2, int(np.sqrt(num_servers / 4)))
        
        # Core switches
        num_core = max(2, num_pods // 2)
        core_switches = list(range(num_core))
        for cs in core_switches:
            G.add_node(cs, label=f"Core_{cs}", type="core")
        
        # Aggregation switches (2 per pod)
        node_id = num_core
        agg_switches = []
        for pod in range(num_pods):
            for agg in range(2):
                agg_id = node_id
                G.add_node(agg_id, label=f"Agg_{pod}_{agg}", type="aggregation", pod=pod)
                # Connect to all core switches
                for cs in core_switches:
                    G.add_edge(cs, agg_id)
                agg_switches.append(agg_id)
                node_id += 1
        
        # ToR switches and servers
        servers_per_tor = max(2, num_servers // (num_pods * 2))
        for pod in range(num_pods):
            for tor_idx in range(2):
                tor_id = node_id
                G.add_node(tor_id, label=f"ToR_{pod}_{tor_idx}", type="tor", pod=pod)
                
                # Connect to aggregation switches in same pod
                agg_base = num_core + pod * 2
                G.add_edge(tor_id, agg_base)
                G.add_edge(tor_id, agg_base + 1)
                
                node_id += 1
                
                # Servers under this ToR
                for srv in range(servers_per_tor):
                    server_id = node_id
                    G.add_node(server_id, label=f"Server_{server_id}", type="server")
                    G.add_edge(tor_id, server_id)
                    node_id += 1
        
        description = f"Data Center TSN Network (Fat-Tree): {G.number_of_nodes()} nodes ({num_pods} pods, {num_servers} servers)"
        return G, description
    
    @staticmethod
    def create_5g_fronthaul_network(num_rrh: int = 20) -> Tuple[nx.Graph, str]:
        """
        5G Fronthaul Network (C-RAN with TSN)
        
        Structure:
        - BBU Pool (Baseband Unit)
        - Fronthaul switches
        - RRH (Remote Radio Heads)
        """
        G = nx.Graph()
        G.name = "5G_Fronthaul_TSN"
        
        # BBU Pool (centralized)
        bbu_pool = 0
        G.add_node(bbu_pool, label="BBU_Pool", type="bbu_pool")
        
        # Fronthaul switches
        num_switches = max(2, num_rrh // 10)
        switches = list(range(1, 1 + num_switches))
        
        for sw in switches:
            G.add_node(sw, label=f"Fronthaul_Switch_{sw}", type="switch")
            G.add_edge(bbu_pool, sw)
        
        # RRHs
        node_id = 1 + num_switches
        rrhs_per_switch = num_rrh // num_switches
        
        for sw in switches:
            for i in range(rrhs_per_switch):
                rrh_id = node_id
                G.add_node(rrh_id, label=f"RRH_{rrh_id}", type="rrh")
                G.add_edge(sw, rrh_id)
                node_id += 1
        
        # Add some mesh connections between switches (redundancy)
        for i in range(len(switches) - 1):
            G.add_edge(switches[i], switches[i+1])
        
        description = f"5G Fronthaul TSN Network: {G.number_of_nodes()} nodes (1 BBU pool, {num_switches} switches, {num_rrh} RRHs)"
        return G, description
    
    @staticmethod
    def save_network(G: nx.Graph, filename: str, path: str = "data/networks/tsn/"):
        """Save network to GML file"""
        import os
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        nx.write_gml(G, filepath)
        print(f"Network saved: {filepath}")
    
    @staticmethod
    def generate_all_tsn_networks():
        """Generate all TSN network types"""
        networks = [
            ("automotive", TSNTopologyGenerator.create_automotive_network(20)),
            ("industrial_ring", TSNTopologyGenerator.create_industrial_network(30, ring=True)),
            ("industrial_line", TSNTopologyGenerator.create_industrial_network(30, ring=False)),
            ("avionics", TSNTopologyGenerator.create_avionics_network(25)),
            ("datacenter", TSNTopologyGenerator.create_datacenter_network(40)),
            ("5g_fronthaul", TSNTopologyGenerator.create_5g_fronthaul_network(20))
        ]
        
        print("\nGenerating TSN Networks:")
        print("="*60)
        
        for name, (G, desc) in networks:
            print(f"\n{name.upper()}:")
            print(f"  {desc}")
            print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            print(f"  Density: {nx.density(G):.4f}")
            if nx.is_connected(G):
                print(f"  Diameter: {nx.diameter(G)}")
            
            TSNTopologyGenerator.save_network(G, f"{name}.gml")
        
        print("\n" + "="*60)
        print("All TSN networks generated successfully!")


if __name__ == "__main__":
    # Generate all TSN networks
    TSNTopologyGenerator.generate_all_tsn_networks()
